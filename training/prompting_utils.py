# coding=utf-8
# Copyright 2024 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
# TODO - SHOULD BE FURTHER IMPROVED.
class UniversalPrompting():
    def __init__(self, text_tokenizer,
                 special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                 max_text_len=8000, max_seq_len=377, ignore_id=-100):
        """
        :param text_tokenizer: original text tokenizer
        """
        self.text_tokenizer = text_tokenizer
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.text_tokenizer.add_tokens(list(special_tokens))
        self.sptids_dict = {token: torch.tensor(self.text_tokenizer.convert_tokens_to_ids([token])) for token in
                            special_tokens}
        self.sptids_dict['<|sot|>'] = torch.tensor([self.text_tokenizer.bos_token_id])
        self.sptids_dict['<|eot|>'] = torch.tensor([self.text_tokenizer.eos_token_id])
        self.sptids_dict['<|pad|>'] = torch.tensor([self.text_tokenizer.pad_token_id])
        # plus 1 because at this time we add a task token before
        self.max_text_len = max_text_len + 1
        self.pad_id = self.text_tokenizer.convert_tokens_to_ids('[PAD]')
        self.ignore_id = ignore_id