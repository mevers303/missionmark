rm static/wordclouds/*
rm output/CountVectorizer.pkl
rm output/NMF.pkl
rm output/TfidfTransformer.pkl
rm output/W.pkl

cp data/govwin_opportunity/pickles/CountVectorizer.pkl output/
cp data/govwin_opportunity/pickles/NMF.pkl output/
cp data/govwin_opportunity/pickles/TfidfTransformer.pkl output/
cp data/govwin_opportunity/pickles/W.pkl output/

cp output/govwin_opportunity/nmf/* static/wordclouds/
