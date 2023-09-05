from utils import Feature, generate_features, create_memo
from preprocess import base_data
# from src.preprocess import base_data
import os
import pandas as pd
import hydra
from sklearn.decomposition import PCA

# 生成された特徴量を保存するパス
Feature.dir = "features"
# trainとtestを結合して基本的な前処理を行ったデータを呼ぶ
data = base_data()

#trainとtestを結合してid欄を削除する。Base_dataについてのメモを残す
class Base_data(Feature):
    def create_features(self):
        self.data = data.drop(columns=["id"])
        create_memo("base_data", "初期")


class Pca(Feature):#特徴量Pcaを作成する。
    def create_features(self):
        n = 20
        #PCAオブジェクトを作成
        pca = PCA(n_components=n)

        #PCAモデルをデータに適合させている。
        #dataという名前のデータフレームから、"train"、"target"、"id" の3つの列を削除した後、PCAを適用
        pca.fit(
            data.drop(
                columns=["train", "target", "id"]
            )
        )
        # この行では、新しいデータフレームのカラム名を生成しています。
        # n_name リストには、"pca_0" から "pca_19" までのカラム名が格納されます
        n_name = [f"pca_{i}" for i in range(n)]
        df_pca = pd.DataFrame(
            pca.transform(data.drop(
                columns=["train", "target", "id"]
            )),
            columns=n_name
        )
        #PCAを適用したデータが self.data に格納される
        self.data = df_pca.copy()
        create_memo("pca", "pcaかけただけ")


#@hydra.main(config_name="../config/config.yaml")
@hydra.main(config_path="../config", config_name="config")
def run(cfg):
    # overwriteがfalseなら上書きはされない
    # globals()からこのファイルの中にある特徴量クラスが選別されてそれぞれ実行される

    generate_features(globals(), cfg.base.overwrite)


# デバッグ用
if __name__ == "__main__":
    run()
