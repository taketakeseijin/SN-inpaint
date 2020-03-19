インターン生：竹内博俊

#Abstract:
もう一方のreadmeは
https://github.com/tadax/glcic
をcloneしたときについてきたものです。
土台はtadaxさんのコードでできていますが、致命的なミスが２つあるので修正済みの私のコードを使うことをお勧めします。
ただし私のコードはSNGANになっています。多少の修正で普通のGANに直すことができますが、多分学習済みモデルを読み込むことはできません。
spectral normalizationの実装については
https://github.com/pfnet-research/chainer-gan-lib/tree/master/common/sn
を参考にしました。

#Requirements:
python3
tensorflow 1.3
opencv
numpy
tqdm

#Contents(added or modified by takeuchi):
/glcic/data
"images" folder which is void
"npy" folder which has "x_test.npy" and "x_train.npy"
試しに動かしやすいように置いておきました。別の内容を学習したい時は中身を変えてください。

/glcic/src
layer_sn.py
network_sn.py
train_sn.py
sn.py
show_epoch.py
show_iteration.py
epoch_log.txt
ite_log.txt
"backup_sn" folder
*_snは基本的にGAN->SNGANに変えたものです。一応logを出力するようになっています。show_*.pyで見てください。
logには自分が学習したときのログが入っています。

/glcic/src/test
demo_sn.py
paint.py
test_sn.py
"paintworks" folder
demo_sn.pyは自分でマスクを作るモードです。


#とりあえずモデルを動かす！:
/glcic/src
にあるbackup_snフォルダをbackupにリネーム
/glcic/src/test
に移動し
python test_sn.py
をすることで
/glcic/src/test/output
に結果が現れます

#とりあえず学習する！:
/glcic/data/images
内に画像をおいてください
/glcic/dataで
python to_npy.py
（これで画像が学習可能な形に変わります）
/glcic/src
にbackupという名前の空フォルダを作る
python train_sn.py
上を実行すると何行かwarningが出た後に
make new epoch_log? Y/N
make new ite_log? Y/N
の二つが出てきますが、とりあえずEnterを押しといてください。
学習が始まります。

#Usage:
基本tadaxさんのコードでできています。tadaxさんのgithubを見るといいでしょう。
以下tadaxさんのコードからの変更追加点等を書きます。

/glcic/data/npyに私が学習で使ったデータをそのまま入れてあります。
http://vis-www.cs.umass.edu/lfw/
から取得したものです。

1.モデルを読み込む
/glcic/src/backup_snに私の学習済みモデルデータがあります。
しかしプログラムがモデルを読み込むときは/glcic/src/backupを参照するので、使用したい場合はリネームしてください

2.ログを読む
一応epoch_log.txt,ite_log.txtを出力します。どちらとも毎エポックごとに更新です。
ite_log.txtが全イテレーションのロスの結果を保存しています。（学習開始の時点で"make new *_log?"にNを返すと前のデータを消さずに追加されます）
epoch_log.txtは全エポックのロス（そのエポック中のロスの総和）を保存しています
python show_epoch.py
python show_iteration.py
でグラフが表示されます。適当に作ったので見やすいように自分で配置等変えてください。
緑がgeneratorで青がdiscriminatorのロスです。どちらもminimizeするように学習しています。

3.モデルをテストする
demo_sn.pyとtest_sn.pyを用意しています。
後者は普通にランダムに矩形でマスクを発生させてimage_completionします。

前者は実行すると画像が表示されると思います。
画像の上で左クリックすることで画像を塗りつぶすことができます。
塗り終わったらキーボードの"s"を押すことで、その部分をマスクとしたimage_completionします。
結果は新たな画面に表示されます。
無限に続くので、やめたくなったらEscキーを押してください。

両者とも/glcic/src/test/outputに結果が保存されます。
