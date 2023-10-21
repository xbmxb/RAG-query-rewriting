from abc import ABC, abstractclassmethod

import torch
from datasets import load_metric
from rl4lms.envs.text_generation.observation import Observation
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rl4lms.envs.text_generation.metric import (
    CIDERMetric,
    MeteorMetric,
    BERTScoreMetric,
    BLEUMetric,
    SpiceMetric,
    ParentToTTo,
    RougeLMax,
    TERMetric,
    chrFmetric,
    IntentAccuracyDailyDialog,
    mmluMetric
)
import numpy as np
from typing import List, Dict, Any
from .llm_utils import llm
from .vicuna_utils import llm as villm
from .myevaluation import f1, ems, hits
from .choiceeval import ems as emsc
from .choiceeval import f1 as f1c
from .choiceeval import hits as hitsc
from .transformersllm import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
import time


class RewardFunction(ABC):
    @abstractclassmethod
    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        """
        Callable for reward functions for text generation

        Args:
            current_observation (Observation): previous observation (s)
            action (int): action performed (a) at s
            next_observation (Observation): observation after the action was performed (s')
            done (bool): whether the episode is finished or not
            meta_info (dict) - other information regarding textual sample
        Returns:
            float: scalar reward
        """
        raise NotImplementedError


class BatchedRewardFunction(ABC):
    """
    Computes rewards for several instances at once
    """

    @abstractclassmethod
    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        An abstract class for batched reward functions for text generation
        """
        raise NotImplementedError


### Automated reward functions ###########################


class CommonGenPenaltyShapingFunction(RewardFunction):
    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            prompt_text = next_observation.prompt_or_input_text
            prefix = "generate a sentence with: "
            concept_n_grams = prompt_text.split(prefix)[1][:-1]

            if (
                concept_n_grams.lower() in next_observation.context_text.lower()
                or prefix in next_observation.context_text.lower()
                or "generate" in next_observation.context_text.lower()
                or "sentence" in next_observation.context_text.lower()
            ):
                penalty_score = -1
            else:
                penalty_score = 0
            return penalty_score
        return 0


class BatchedCommonGenPenaltyShapingFunction(BatchedRewardFunction):
    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_info: Dict[str, Any] = None,
    ) -> List[float]:
        scores = []
        for done, prompt_text, gen_text in zip(dones, prompt_texts, gen_texts):
            if done:
                prefix = "generate a sentence with: "
                concept_n_grams = prompt_text.split(prefix)[1][:-1]

                if (
                    concept_n_grams.lower() in gen_text.lower()
                    or prefix in gen_text.lower()
                    or "generate" in gen_text.lower()
                    or "sentence" in gen_text.lower()
                ):
                    penalty_score = -1
                else:
                    penalty_score = 0
                scores.append(penalty_score)
        return scores


class MeteorRewardFunction(RewardFunction):
    def __init__(self, shaping_fn: str = None) -> None:
        super().__init__()
        self._metric = MeteorMetric()
        from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

        # compute meteor at the end of episode
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_dict = self._metric.compute(None, predicted, references)
            score = metric_dict["lexical/meteor"][1]

            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                score = score + aux_score
            return score
        return 0


class RougeRewardFunction(RewardFunction):
    def __init__(
        self, rouge_type: str, shaping_fn: str = None, use_single_ref: bool = True
    ) -> None:
        super().__init__()
        self._metric = load_metric("rouge")
        self._rouge_type = rouge_type
        from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )
        self._use_single_ref = use_single_ref

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            # TBD: considers only one reference for now
            if self._use_single_ref:
                references = [next_observation.target_or_reference_texts[0]]
            else:
                references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]

            metric_results = self._metric.compute(
                predictions=predicted, references=references, use_stemmer=True
            )
            reward = metric_results[self._rouge_type].mid.fmeasure
            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                reward = reward + aux_score
            return reward
        return 0

class LLMRewardFunction(RewardFunction):
    def __init__(
        self, pid: int = 0, shaping_fn: str = None, use_single_ref: bool = True, weight_f1: float = None, norm: bool = False, searchfunc = 'plain', topn=None, max_words_perdoc=None
    ) -> None:
        super().__init__()
        # self._f1 = load_metric("f1")
        # self._em = load_metric("exact_match")
        self._weight_f1 = weight_f1 if weight_f1 else 1.0
        self._pid = pid
        self._norm = norm
        from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )
        self._use_single_ref = use_single_ref

        self.searchfunc = searchfunc
        self.topn = topn
        self.max_words_perdoc = max_words_perdoc

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        # jec dataset
        if done:
            # TBD: considers only one reference for now
            if self._use_single_ref:
                references = [next_observation.target_or_reference_texts[0]]
            else:
                references = [next_observation.target_or_reference_texts]
            predicted_query = [next_observation.context_text]
            # search + prompt
            questions_ = next_observation.prompt_or_input_text # one eg
            # print(questions_)
            # prefix = "rewrite a better search query: "
            # questions = [questions_.split(prefix)[1]]
            questions = [questions_]
            # print(questions)
            # print(predicted_query)
            predicted = llm(
                queries = predicted_query,
                questions = questions,
                pid = self._pid,
                bar = False,
                searchfunc = self.searchfunc, topn=self.topn, max_words_perdoc=self.max_words_perdoc
            )
            # predicted = list(predicted)
            # print("returned predcitions: ", predicted)
            # print("references: ", references)
            metric_results_f1 = []
            metric_results_em = []
            for i, (p, r) in enumerate(zip(predicted, references)):
                metric_results_em.append(emsc(p[0], r))
                # metric_results_f1.append(f1(p[0], r))
            reward = sum(metric_results_em)/len(metric_results_em) # + sum(metric_results_f1)/len(metric_results_f1) * self._weight_f1
            # if self._norm:
            #     reward = reward / (1 + self._weight_f1)
            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                reward = reward + aux_score
            print(reward)
            return reward
        return 0

class LLMRewardFunctionv2(RewardFunction):
    def __init__(
        self, pid: int = 0, shaping_fn: str = None, use_single_ref: bool = True, weight_f1: float = None
    ) -> None:
        super().__init__()
        # self._f1 = load_metric("f1")
        # self._em = load_metric("exact_match")
        self._weight_f1 = weight_f1 if weight_f1 else 1.0
        self._pid = pid
        from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )
        self._use_single_ref = use_single_ref

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            # TBD: considers only one reference for now
            if self._use_single_ref:
                references = [next_observation.target_or_reference_texts[0]]
            else:
                references = [next_observation.target_or_reference_texts]
            predicted_query = [next_observation.context_text]
            # search + prompt
            questions_ = next_observation.prompt_or_input_text # one eg
            # print(questions_)
            prefix = "rewrite a better search query: "
            questions = [questions_.split(prefix)[1]]
            # print(questions)
            # print(predicted_query)
            predicted = llm(
                queries = predicted_query,
                questions = questions,
                pid = self._pid,
                bar = False 
            )
            metric_results_f1 = []
            metric_results_em = []
            for i, (p, r) in enumerate(zip(predicted, references)):
                metric_results_em.append(ems(p[0], r))
                metric_results_f1.append(f1(p[0], r))
            # em = sum(metric_results_em)/len(metric_results_em)
            # f1 = sum(metric_results_f1)/len(metric_results_f1)
            reward_ = []
            for e, f in zip(metric_results_em, metric_results_f1):
                # if e: #em = 1
                #     r = e + f * self._weight_f1
                # else: #em = 0
                #     r = (f - 1) * self._weight_f1
                # # norm
                # r = (r+1)/(2+self._weight_f1)
                r = e
                reward_.append(r)
            reward =  sum(reward_)/len(reward_)
            print(reward) # multichoice
            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                reward = reward + aux_score
            return reward
        return 0

class LLMRewardFunctionv3(RewardFunction):
    def __init__(
        self, pid: int = 0, shaping_fn: str = None, use_single_ref: bool = True, weight_f1: float = None, norm: bool = False, think:bool = False, max_obs = None,searchfunc = 'plain', topn=None, max_words_perdoc=None
    ) -> None:
        super().__init__()
        # self._f1 = load_metric("f1")
        # self._em = load_metric("exact_match")
        self._weight_f1 = weight_f1 if weight_f1 else 1.0
        self._pid = pid
        self._norm = norm
        self._think = think
        self._max_obs = max_obs
        from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )
        self._use_single_ref = use_single_ref
        self.searchfunc = searchfunc
        self.topn = topn
        self.max_words_perdoc = max_words_perdoc

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            # add an broadcast
            
            # TBD: considers only one reference for now
            if self._use_single_ref:
                references = [next_observation.target_or_reference_texts[0]]
            else:
                references = [next_observation.target_or_reference_texts]
            predicted_query = [next_observation.context_text]
            # search + prompt
            questions_ = next_observation.prompt_or_input_text # one eg
            # print(questions_)
            prefix = "rewrite a better search query: "
            questions = [questions_.split(prefix)[1]]
            # print(questions)
            # print(predicted_query)
            predicted, inlines = llm(
                queries = predicted_query,
                questions = questions,
                pid = self._pid,
                bar = False,
                think = self._think,
                max_obs = self._max_obs,
                searchfunc = self.searchfunc, topn=self.topn, max_words_perdoc=self.max_words_perdoc
            )
            # predicted = list(predicted)
            # print("returned predcitions: ", predicted)
            # print("references: ", references)
            metric_results_f1 = []
            metric_results_em = []
            metric_hit = []
            for i, (p, r) in enumerate(zip(predicted, references)):
                metric_results_em.append(ems(p[0], r))
                metric_results_f1.append(f1(p[0], r))
            for i, (inl, r) in enumerate(zip(inlines, references)):
                metric_hit.append(hits(r, inl['output'], dn=0, dl=False))
            reward_qa = sum(metric_results_em)/len(metric_results_em) + sum(metric_results_f1)/len(metric_results_f1) * self._weight_f1
            reward_retr = sum(metric_hit)/len(metric_hit)
            if self._norm:
                reward_qa = reward_qa / (1 + self._weight_f1)
            reward = reward_qa + reward_retr
            print(reward, reward_qa, reward_retr)
            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                reward = reward + aux_score
            return reward
        return 0

class mmluRewardFunction(RewardFunction):
    def __init__(
        self, pid: int = 0, shaping_fn: str = None, use_single_ref: bool = True, weight_f1: float = None, norm: bool = False, think:bool = False, max_obs = None,searchfunc = 'plain', topn=None, max_words_perdoc=None, black=None
    ) -> None:
        try:
            super().__init__()
            # self._f1 = load_metric("f1")
            # self._em = load_metric("exact_match")
            self._weight_f1 = weight_f1 if weight_f1 else 1.0
            self._pid = pid
            self._norm = norm
            self._think = think
            self._max_obs = max_obs
            
            from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

            self._shaping_fn = (
                RewardFunctionRegistry.get(shaping_fn, {})
                if shaping_fn is not None
                else shaping_fn
            )
            self._use_single_ref = use_single_ref
            self.searchfunc = searchfunc
            self.topn = topn
            self.max_words_perdoc = max_words_perdoc
            self.engine = '/xinbei_data/replug/baseline_new/transformers/examples/legacy/seq2seq/vicuna13_recovered' # fixed 13b vicuna
            self.config = LlamaConfig.from_pretrained(self.engine)
            self.tokenizer = LlamaTokenizer.from_pretrained(self.engine)
            self._last_gpu = f"cuda:{torch.cuda.device_count() - 1}"
            print('loading vicuna: ', self.engine)
            self.starttime = time.time()
            # self.llama = LlamaForCausalLM.from_pretrained(
            #     self.engine,
            #     torch_dtype=torch.float16, low_cpu_mem_usage=True,
            #     device_map = {'':3}
            #     )
            self.llama = None
            # self.llama = mmluMetric(pid, use_single_ref, think, max_obs, searchfunc, topn, max_words_perdoc, black)
            self.endtime = time.time()
            print('llm loaded: '+ str(self.endtime-self.starttime))
            if torch.cuda.is_available():
                self.device = torch.device(self._last_gpu)
            else:
                self.device = torch.device('cpu')
            print("using device: ", self.device)
            # self.llama.to(self.device)

            if black:
                self.black  = []
                with open(black,'r') as f:
                    for line in f:
                        line = line.split("\\")[-1]
                        self.black.append(line.strip())
            else:
                self.black = None
        except Exception as e:
            from loguru import logger
            logger.e(e)

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            # add an broadcast
            
            # TBD: considers only one reference for now
            if self._use_single_ref:
                references = [next_observation.target_or_reference_texts[0]]
            else:
                references = [next_observation.target_or_reference_texts]
            predicted_query = [next_observation.context_text]
            # search + prompt
            questions_ = next_observation.prompt_or_input_text # one eg
            # print(questions_)
            prefix = "rewrite a better search query: "
            questions = [questions_.split(prefix)[1]]
            # print(questions)
            # print(predicted_query)
            predicted, inlines = villm(
                model = self.llama,
                device = self.device,
                tokenizer = self.tokenizer,
                black = self.black,
                queries = predicted_query,
                questions = questions,
                pid = self._pid,
                bar = False,
                think = self._think,
                max_obs = self._max_obs,
                searchfunc = self.searchfunc, topn=self.topn, max_words_perdoc=self.max_words_perdoc
            )
            # predicted = list(predicted)
            # print("returned predcitions: ", predicted)
            # print("references: ", references)
            metric_results_f1 = []
            metric_results_em = []
            metric_hit = []
            for i, (p, r) in enumerate(zip(predicted, references)):
                metric_results_em.append(emsc(p[0], r))
                # metric_results_f1.append(f1c(p[0], r))
            # for i, (inl, r) in enumerate(zip(inlines, references)):
            #     metric_hit.append(hitsc(r, inl['output'], dn=0, dl=False))
            reward_qa = sum(metric_results_em)/len(metric_results_em) # + sum(metric_results_f1)/len(metric_results_f1) * self._weight_f1
            # reward_retr = sum(metric_hit)/len(metric_hit)
            # if self._norm:
            #     reward_qa = reward_qa / (1 + self._weight_f1)
            reward = reward_qa # + reward_retr
            print(reward, reward_qa)
            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                reward = reward + aux_score
            return reward
        return 0
    
class RougeCombined(RewardFunction):
    def __init__(self, shaping_fn: str = None) -> None:
        super().__init__()
        self._metric = load_metric("rouge")
        from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            # TBD: considers only one reference for now
            references = [next_observation.target_or_reference_texts[0]]
            predicted = [next_observation.context_text]

            metric_results = self._metric.compute(
                predictions=predicted, references=references, use_stemmer=True
            )

            rouge_keys = ["rouge1", "rouge2", "rougeL"]
            scores = [
                metric_results[rouge_type].mid.fmeasure for rouge_type in rouge_keys
            ]
            reward = np.mean(scores)
            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                reward = reward + aux_score
            return reward
        return 0


class BERTScoreRewardFunction(RewardFunction):
    def __init__(self, language: str = "en") -> None:
        super().__init__()
        self._metric = BERTScoreMetric(language)

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(None, predicted, references)
            bert_score = metric_results["semantic/bert_score"][1]
            return bert_score
        return 0


class BLEURewardFunction(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = BLEUMetric()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(None, predicted, references)
            bleu_score = metric_results["lexical/bleu"][1]
            return bleu_score
        return 0


class SacreBleu(RewardFunction):
    def __init__(self, **args) -> None:
        super().__init__()
        self._metric = load_metric("sacrebleu")
        self._args = args

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(
                predictions=predicted, references=references, **self._args
            )
            return metric_results["score"] / 100
        return 0


class SpiderRewardFunction(BatchedRewardFunction):
    def __init__(
        self, spice_coeff: float, cider_coeff: float, shaping_fn: str = None
    ) -> None:
        """
        Spice + Cider
        """
        super().__init__()
        self._spice_metric = SpiceMetric()
        self._cider_metric = CIDERMetric()
        self._spice_coeff = spice_coeff
        self._cider_coeff = cider_coeff
        from rl4lms.envs.text_generation.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_info: Dict[str, Any] = None,
    ) -> List[float]:
        prompts = []
        gens = []
        refs = []
        indices_with_done = []
        rewards = torch.zeros(len(prompt_texts))
        for ix, (prompt, gen, ref, done) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts, dones)
        ):
            if done:
                prompts.append(prompt)
                gens.append(gen)
                refs.append(ref)
                indices_with_done.append(ix)

        if len(indices_with_done) > 0:
            spice_scores = self._spice_metric.compute(prompts, gens, refs)[
                "lexical/spice"
            ][0]
            cider_scores = self._cider_metric.compute(prompts, gens, refs)[
                "lexical/cider"
            ][0]
            total_scores = self._spice_coeff * np.array(
                spice_scores
            ) + self._cider_coeff * np.array(cider_scores)

            if self._shaping_fn is not None:
                aux_scores = self._shaping_fn(prompt_texts, gen_texts, ref_texts, dones)
            else:
                aux_scores = [0] * len(indices_with_done)

            for ind, score, aux_score in zip(
                indices_with_done, total_scores, aux_scores
            ):
                rewards[ind] = score + aux_score

        return rewards


#############################################################################

########## Learned Reward Functions##########################################


class LearnedRewardFunction(RewardFunction):
    def __init__(
        self, model_name: str, label_ix: int, include_prompt_for_eval: bool = True
    ) -> None:
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._metric_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._metric_tokenizer.truncation_side = "left"
        self._metric_model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self._device)
        self._label_ix = label_ix
        self._include_prompt_for_eval = include_prompt_for_eval

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            generated_text = (
                current_observation.prompt_or_input_text
                if self._include_prompt_for_eval
                else ""
            )
            generated_text += next_observation.context_text

            with torch.no_grad():
                encoded = self._metric_tokenizer(
                    generated_text, return_tensors="pt", truncation=True, padding=True
                )
                outputs = self._metric_model(
                    input_ids=encoded.input_ids.to(self._device),
                    attention_mask=encoded.attention_mask.to(self._device),
                )
                scores = torch.softmax(outputs.logits.flatten(), dim=0)
                score = scores[self._label_ix].item()
                return score
        return 0


# class BLEURTRewardFunction(RewardFunction):
#     def __init__(self, checkpoint: str = None):
#         super().__init__()
#         self._metric = load_metric("bleurt", checkpoint=checkpoint)

#     def __call__(
#         self,
#         current_observation: Observation,
#         action: int,
#         next_observation: Observation,
#         done: bool,
#         meta_info: Dict[str, Any] = None,
#     ) -> float:
#         if done:
#             references = [next_observation.target_or_reference_texts]
#             predicted = [next_observation.context_text]
#             metric_results = self._metric.compute(
#                 predictions=predicted, references=references
#             )
#             score = metric_results["scores"][0]
#             return score
#         return 0


class PARENTRewardFunction(RewardFunction):
    """
    PARENT F1 score as the reward
    """

    def __init__(self) -> None:
        super().__init__()
        self._metric = ParentToTTo()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            generated_texts = [next_observation.context_text]
            meta_infos = [meta_info]
            scores = self._metric.compute(None, generated_texts, None, meta_infos)
            reward = scores["table_to_text/parent_overall_f_score"][0][0]
            return reward
        return 0


class RougeLMaxRewardFunction(RewardFunction):
    def __init__(self, **args) -> None:
        super().__init__()
        self._metric = RougeLMax(**args)

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            meta_infos = [meta_info]
            scores = self._metric.compute(None, predicted, references, meta_infos)
            reward = scores["lexical/rouge_l_max"][0][0]
            return reward
        return 0


class TER(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = TERMetric()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

        # compute score at the end of episode
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_dict = self._metric.compute(None, predicted, references)
            score = metric_dict["lexical/ter"][1]
            score = 1 - score
            return score
        return 0


class chrF(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = chrFmetric()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

        # compute score at the end of episode
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_dict = self._metric.compute(None, predicted, references)
            score = metric_dict["lexical/chrf"][1]
            return score
        return 0


class IntentAccuracy(BatchedRewardFunction):
    def __init__(
        self, shape: bool = True, intent_coeff: float = 1.0, auto_coeff: float = 1.0
    ) -> None:
        super().__init__()
        self._metric = None
        self._shape = shape
        self._intent_coeff = intent_coeff
        self._auto_coeff = auto_coeff
        self._shaping_metric = MeteorMetric()

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:

        if self._metric is None:
            self._metric = IntentAccuracyDailyDialog()

        # compute rewards for finished episodes only
        rewards = np.zeros(len(gen_texts))

        done_prompt_texts = []
        done_gen_texts = []
        done_ref_texts = []
        done_meta_infos = []
        done_ixs = []
        for ix, (prompt, gen, ref, meta_info, done) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts, meta_infos, dones)
        ):
            if done:
                done_prompt_texts.append(prompt)
                done_gen_texts.append(gen)
                done_ref_texts.append(ref)
                done_meta_infos.append(meta_info)
                done_ixs.append(ix)

                if self._shape:
                    score = self._shaping_metric.compute(
                        done_prompt_texts, done_gen_texts, done_ref_texts
                    )
                    rewards[ix] = self._auto_coeff * score["lexical/meteor"][1]

        scores = self._metric.compute(
            done_prompt_texts, done_gen_texts, done_ref_texts, done_meta_infos
        )["intent/accuracy"][0]
        rewards[done_ixs] += self._intent_coeff * np.array(scores)
        return rewards.tolist()


if __name__ == "__main__":
    predictions = "hello there general kenobi"
    references = ["hello there general kenobi", "hello there!!"]
    observation = Observation(
        None, None, None, None, None, predictions, references, None, None, None, None
    )

    reward_fn = MeteorRewardFunction()
    print(reward_fn(None, None, observation, True))

    reward_fn = chrF()
    print(reward_fn(None, None, observation, True))

    reward_fn = RougeCombined()
    print(reward_fn(None, None, observation, True))

    reward_fn = RougeRewardFunction(rouge_type="rouge1")
    print(reward_fn(None, None, observation, True))

    reward_fn = RougeRewardFunction(rouge_type="rouge2")
    print(reward_fn(None, None, observation, True))

    reward_fn = RougeRewardFunction(rouge_type="rougeL")
    print(reward_fn(None, None, observation, True))

    reward_fn = BERTScoreRewardFunction(language="en")
    print(reward_fn(None, None, observation, True))

    reward_fn = BLEURewardFunction()
    print(reward_fn(None, None, observation, True))

    # reward_fn = BLEURTRewardFunction()
    # print(reward_fn(None, None, observation, True))
