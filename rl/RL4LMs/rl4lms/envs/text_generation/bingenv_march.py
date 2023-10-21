import ast
import json
import time, os
import gym
import requests
from bs4 import BeautifulSoup
from .bing import searchbing
from .bing_utils.bing import searchsele, searchbl, searchr1
from .myevaluation import normalize_answer

# import wikipedia

def clean_str(p):
  # print
  try:
    p = p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    return p
  except:
    return ''


class textSpace(gym.spaces.Space):
  def contains(self, x) -> bool:
    """Return boolean specifying if x is a valid member of this space."""
    return isinstance(x, str)


class BingEnv(gym.Env):

  def __init__(self):
    """
      Initialize the environment.
    """
    super().__init__()
    self.page = None  # current Wikipedia page
    self.obs = None  # current observation
    self.lookup_keyword = None  # current lookup keyword
    self.lookup_list = None  # list of paragraphs containing current lookup keyword
    self.lookup_cnt = None  # current lookup index
    self.steps = 0  # current number of steps
    self.answer = None  # current answer from the agent
    self.observation_space = self.action_space = textSpace()
    self.search_time = 0
    self.num_searches = 0
    self.appendsimilar = False
    
  def _get_obs(self):
    return self.obs

  def _get_info(self):
    return {"steps": self.steps, "answer": self.answer}

  def reset(self, seed=None, return_info=False, options=None):
    # We need the following line to seed self.np_random
    # super().reset(seed=seed)
    self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                "finish[].\n")
    self.page = None
    self.lookup_keyword = None
    self.lookup_list = None
    self.lookup_cnt = None
    self.steps = 0
    self.answer = None
    observation = self._get_obs()
    info = self._get_info()
    return (observation, info) if return_info else observation

  def construct_lookup_list(self, keyword):
    # find all paragraphs
    if self.page is None:
      return []
    paragraphs = self.page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]

    parts = sentences
    parts = [p for p in parts if keyword.lower() in p.lower()]
    return parts

  @staticmethod
  def get_page_obs(page):
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    return ' '.join(sentences[:5])

    # ps = page.split("\n")
    # ret = ps[0]
    # for i in range(1, len(ps)):
    #   if len((ret + ps[i]).split(" ")) <= 50:
    #     ret += ps[i]
    #   else:
    #     break
    # return ret

  def search_step(self, entity, use_en ):
    # entity_ = normalize_answer(entity).replace(" ", "+")
    entity_ = normalize_answer(entity)
    # search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    old_time = time.time()
    # response_text = requests.get(search_url).text
    # print(response_text)
    print(entity_)
    if entity == 'none':
      self.obs = ''
      return
    response_json = searchbing(entity_)
    self.search_time += time.time() - old_time
    self.num_searches += 1
    res_text = []
    hint = " Hint: "
    if use_en and 'entities' in response_json.keys() and 'value' in response_json['entities'].keys():
      for en in response_json['entities']['value']:
        if 'description' in en.keys():
          res_text.append(en['description'])
        if 'name' in en.keys():
          hint += en['name']+' '
    if 'webPages' in response_json.keys() and 'value' in response_json['webPages'].keys() and len(res_text)==0:
      for en in response_json['webPages']['value']:
        if 'name' in en.keys() and 'snippet' in en.keys():
          res_text.append(' '.join([en['name'], en['snippet']]))
    if len(res_text) == 0:
      print(response_json)
      # os._exit()
      res_text = ['']
    self.obs = ' '.join(res_text)
    if hint != " Hint: ":
      self.obs += hint[:-1]
    self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
    # if result_divs:  # mismatch
    #   self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
    #   self.obs = f"Similar: {self.result_titles[:5]}."
    #   if self.appendsimilar and self.num_searches<3:
    #     self.search_step(self.result_titles[0])
    #     entitys = [ i.lower() for i in entity.split()]
    #     res1 = [ i.lower() for i in self.result_titles[0].split()]
    #     entity_look = [i for i in entitys if i not in res1]
    #     lookupres = []
    #     for _ in range(3):
    #       self.step('lookup['+entity_look[0]+']')
    #       if self.obs.startswith("No more results"):
    #         break
    #       # return all lookuplists
    #       lookupres.append(self.obs)
    #     self.obs = f"Similar: {self.result_titles[:5]}." + ' '.join(lookupres)
    #     # if self.obs.startswith("No more results."):
    #     #   self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
    # else:
    #   page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
    #   # print('-------page--------')
    #   # print(page)
    #   if any("may refer to:" in p for p in page):
    #     self.search_step("[" + entity + "]")
    #   else:
    #     self.page = ""
    #     for p in page:
    #       if len(p.split(" ")) > 2:
    #         self.page += clean_str(p)
    #         if not p.endswith("\n"):
    #           self.page += "\n"
    #     self.obs = self.get_page_obs(self.page)
    #     self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

  
  def step(self, action, use_en, func, gold, topn=None, max_words_perdoc=None):
    reward = 0
    done = False
    action = action.strip()
    if self.answer is not None:  # already finished
      done = True
      print('return ealier')
      return self.obs, reward, done, self._get_info()
    
    if action.startswith("search[") and action.endswith("]"):
      entity = action[len("search["):-1]
      # entity_ = entity.replace(" ", "_")
      # search_url = f"https://en.wikipedia.org/wiki/{entity_}"
      self.search_step(entity, use_en)
    elif action.startswith("lookup[") and action.endswith("]"):
      keyword = action[len("lookup["):-1]
      if self.lookup_keyword != keyword:  # reset lookup
        self.lookup_keyword = keyword
        self.lookup_list = self.construct_lookup_list(keyword)
        self.lookup_cnt = 0
      if self.lookup_cnt >= len(self.lookup_list):
        self.obs = "No more results.\n"
      else:
        self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
        self.lookup_cnt += 1
    elif action.startswith("finish[") and "]" in action:
      # answer = action[len("finish["):-1]
      answer = action.split('[')[1].split(']')[0]
      print(answer)
      self.answer = answer
      done = True
      self.obs = f"Episode finished, reward = {reward}\n"
    elif action.startswith("think[") and action.endswith("]"):
      self.obs = "Nice thought."
    else:
      self.obs = "Invalid action: {}".format(action)

    self.steps += 1

    return self.obs, reward, done, self._get_info()
  
  def get_time_info(self):
    speed = self.search_time / self.num_searches if self.num_searches else 0
    return {
        "call_speed": speed,
        "call_time": self.search_time,
        "num_calls": self.num_searches,
    }
