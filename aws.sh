cd src
# python data.py
python vectorizer.py
python nlp.py -t 100
cd ..
bash migrate_to_webapp.sh
# sudo shutdown -hP now
