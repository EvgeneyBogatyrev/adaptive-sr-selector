from catboost import CatBoostRegressor
from feature_extractor import FeatureExtractor
from utils import get_sr_list


class SRSelector():

    def __init__(self, model_path="weights/best.cbm", features="all", frame_step=1, tmp_dir="./tmp", split_to_scenes=True):
        self.model = CatBoostRegressor(iterations=200,
                learning_rate=0.05,
                depth=5,
                loss_function="LogLinQuantile",
                silent=True)
        
        self.model.load_model(model_path)

        self.sr_list = get_sr_list()

        self.feature_extractor = FeatureExtractor(features=features, frame_step=frame_step, 
                                                    tmp_dir=tmp_dir, split_to_scenes=split_to_scenes)


    def __call__(self, video_path, top_k=3):
        overall_features = self.feature_extractor(video_path)
        overall_results = []

        for features in overall_features:
            feature_list = []
            for feat_name in features.keys():
                if feat_name == "bitrate":
                    feature_list.append(features[feat_name])
                else:
                    for quality in ['min', 'max', 'mean']:
                        feature_list.append(features[feat_name][quality])

            results = {}
            for sr_name in self.sr_list:
                tmp_feature_list = [sr_name] + feature_list
                score = self.model.predict(tmp_feature_list)
                results[sr_name] = score

            data = zip(list(results.keys()), list(results.values()))
            data = sorted(data, key=lambda x: x[1], reverse=True)

            overall_results.append([x[0] for x in data[:top_k]])
        return overall_results









