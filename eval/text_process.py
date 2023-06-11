import cfg
import numpy as np

# 根据四边形顶点计算四边形面积（自动修正顺逆时针顶点顺序）
def calc_polygon_area(polygon):
    coord = np.array(polygon).reshape((4,2))
    temp_det = 0
    for idx in range(3):
        temp = np.array([coord[idx],coord[idx+1]])
        temp_det += np.linalg.det(temp)
    temp_det += np.linalg.det(np.array([coord[-1],coord[0]]))
    return abs(temp_det*0.5)

class TextProcess(object):
    # 计算实体对象总分
    def calc_total_score(self, text_obj):
        words = text_obj['words']
        words_char_score = text_obj['words_char_score']
        polygon = text_obj['polygon']
        polygon_score = text_obj['polygon_score']
        return np.sum(polygon_score) * cfg.polygon_vertex_power \
                + calc_polygon_area(polygon) * cfg.polygon_area_power

    def find_enterprise(self, text_objects):
        ans_text_obj = None
        ans_score = 0
        for text_obj in text_objects:
            if 'words' not in text_obj or 'polygon' not in text_obj: # 合法性检查
                continue
            words = text_obj['words']
            words = words.strip() # 去除首尾空字符
            if len(words) == 0 or '#' in words: # 若字段存在无效字符，则直接跳过
                continue
            if len(words) > cfg.enterprise_len_threshold: # 若字段长度超出企业实体命名阈值，则直接跳过
                continue
            
            if ans_text_obj is None:
                ans_text_obj = text_obj
                ans_score = self.calc_total_score(text_obj)
            else:
                this_score = self.calc_total_score(text_obj) # 计算当前得分
                if this_score > ans_score: # 如果当前得分比最大得分大则选取得分大的作为结果
                    ans_text_obj = text_obj
                    ans_score = this_score
                
        if ans_text_obj is not None:
            return ans_text_obj['words'].strip(), ans_text_obj['polygon']
        return None, None