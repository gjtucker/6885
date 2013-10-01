python locu_classifier.py models/classifier.pkl
python locu_matching.py models/classifier.pkl models/matcher.pkl
python locu_predict.py models/matcher.pkl locu_test_hard.json foursquare_test_hard.json leaderboard_matching.csv
