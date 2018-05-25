rm static/wordclouds/*
rm static/CountVectorizer.pkl
rm static/NMF.pkl
rm static/TfidfTransformer.pkl
rm static/W.pkl

cp data/govwin_opportunity/pickles/CountVectorizer.pkl static/
cp data/govwin_opportunity/pickles/NMF.pkl static/
cp data/govwin_opportunity/pickles/TfidfTransformer.pkl static/
cp data/govwin_opportunity/pickles/W.pkl static/

cp output/govwin_opportunity/nmf/* static/wordclouds/
