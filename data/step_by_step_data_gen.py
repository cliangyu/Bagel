import json
import os
import traceback
from PIL import Image, ImageFile, PngImagePlugin

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class StepByStepGenerationDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform,
        jsonl_path_list, data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        shuffle_lines=False, shuffle_seed=0,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status

        self.start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')
        self.im_start = tokenizer.convert_tokens_to_ids('<|im_start|>')

        self.data_paths = self.get_data_paths(
            jsonl_path_list,
            data_dir_list,
            num_used_data,
            shuffle_lines,
            shuffle_seed,
        )
        self.set_epoch()

    def get_data_paths(
        self,
        jsonl_path_list,
        data_dir_list,
        num_used_data,
        shuffle_lines,
        shuffle_seed,
    ):
        data_paths = []
        for jsonl_path, image_dir, num_data_point in zip(
            jsonl_path_list, data_dir_list, num_used_data
        ):
            with open(jsonl_path, 'r') as f:
                raw_data = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            raw_data = raw_data[:num_data_point]
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])
        return data_paths

    def _init_data(self):
        """Initialize the data structure"""
        data = {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            'num_tokens': 0,
        }
        return data

    def _add_text(self, data, text, need_loss, enable_cfg=True, next_token_label=None):
        """Add text"""
        text_ids = self.tokenizer.encode(text)
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)

        # If next_token_label is provided, the im_end token should predict it
        special_token_loss = 1 if next_token_label is not None else 0

        data['sequence_plan'].append(
            {
                'type': 'text',
                'enable_cfg': int(enable_cfg),
                'loss': int(need_loss),
                'special_token_loss': special_token_loss,
                'special_token_label': next_token_label,
            }
        )
        return data

    def _add_image_for_generation(self, data, image, need_loss, enable_cfg=0):
        """Add image for generation"""
        data['sequence_plan'].append({
            'type': 'vae_image',
            'enable_cfg': enable_cfg,
            'loss': int(need_loss),
            'special_token_loss': 0,
            'special_token_label': None,
        })

        image_tensor = self.transform(image)
        height, width = image_tensor.shape[1:]
        data['num_tokens'] += width * height // self.transform.stride ** 2
        data['image_tensor_list'].append(image_tensor)
        return data

    def _add_image_for_understanding(self, data, image, enable_cfg=1):
        """Add image for understanding, as condition. Teacher-forcing"""
        data['sequence_plan'].append({
            'type': 'vit_image',
            'enable_cfg': enable_cfg,
            'loss': 0,
            'special_token_loss': 0,
            'special_token_label': None,
        })

        vit_image_tensor = self.vit_transform(image)
        height, width = vit_image_tensor.shape[1:]
        data['num_tokens'] += width * height // self.vit_transform.stride ** 2
        data['image_tensor_list'].append(vit_image_tensor)
        return data

    def parse_step_by_step_data(self, data_item, image_dir):
        """
        parse step-by-step data, to construct interleaved sequence

        Key design:
        1. For the first image: generation, no understanding
        2. Sequent images: previous image as condition, current image as generation

        Sequence structure:
        Human prompt →
        Step1 text → Image1_gen (gen target) →
        Step2 text → Image1_understand (und target) → Image2_gen (gen target) →
        Step3 text → Image2_understand (und target) → Image3_gen (gen target)
        """
        prompt = "You are an AI reasoning assistant specialized in creating images step by step. When given a prompt, you should break it down into logical steps and generate intermediate images that progressively build toward the final result. Each step should have a clear description of what you're adding or modifying. Continue this text-image pattern until the final image is complete."

        data = self._init_data()

        data = self._add_text(data, prompt, need_loss=False, enable_cfg=True)

        try:
            # 1. Human prompt (no loss calculation)
            human_conversation = [conv for conv in data_item['conversations'] if conv['from'] == 'human'][0]
            data = self._add_text(data, human_conversation['value'], need_loss=False, enable_cfg=0)

            # 2. interleaved output sequence
            gpt_conversations = [conv for conv in data_item['conversations'] if conv['from'] == 'gpt']

            for i, conversation in enumerate(gpt_conversations):
                if i > 0:
                    prev_image_path = os.path.join(image_dir, data_item['images'][i-1])
                    prev_raw_image = pil_img2rgb(Image.open(prev_image_path))
                    data = self._add_image_for_understanding(
                        data, prev_raw_image, enable_cfg=1
                    )
                # Add text (loss)
                step_text = conversation['value']
                step_text = step_text.replace('<image>', '').replace('\n\n', '\n').strip()

                if i < len(data_item['images']):
                    next_token_label = self.start_of_image

                data = self._add_text(data, step_text, need_loss=True, enable_cfg=True, next_token_label=next_token_label)

                if i < len(data_item['images']):
                    current_image_path = os.path.join(image_dir, data_item['images'][i])
                    current_raw_image = pil_img2rgb(Image.open(current_image_path))
                    data = self._add_image_for_generation(
                        data, current_raw_image, need_loss=True, enable_cfg=0
                    )
        except Exception as e:
            print(f'Error parsing step-by-step data: {e}')
            return {}

        return data

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (json_line, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                try:
                    data_item = json.loads(json_line.strip())

                    data = self.parse_step_by_step_data(data_item, image_dir)

                    if len(data) == 0:
                        continue

                    data['data_indexes'] = {
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    }

                    yield data
                except Exception as e:
                    print(f'Error processing row {row_idx}: {e}')
                    traceback.print_exc()
                    continue

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
