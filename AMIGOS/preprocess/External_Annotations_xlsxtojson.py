import sys
import pandas as pd
import json

if __name__ == "__main__":
    filepath = sys.argv[1]
    file = pd.read_excel(filepath)
    file['Valence_Annotator_3'] = file['Valence_Annotator_3'].apply(lambda x: min(1, x))
    file['Arousal_Annotator_2'] = file['Arousal_Annotator_2'].apply(lambda x: max(-1, x))
    data = dict()
    for element in file.iterrows():
        arousal_anot = element[1][['Arousal_Annotator_1', 'Arousal_Annotator_2', 'Arousal_Annotator_3']]
        valence_anot = element[1][['Valence_Annotator_1', 'Valence_Annotator_2', 'Valence_Annotator_3']]
        if element[1]['UserID'] not in data:
            data[element[1]['UserID']] = {'Videos': dict()}
        usr_vids = data[element[1]['UserID']]['Videos']
        if element[1]['VideoID'] not in usr_vids:
            usr_vids[element[1]['VideoID']] = {'Segments': dict()}
        usr_segs = usr_vids[element[1]['VideoID']]['Segments']
        if element[1]['Segment_Index'] not in usr_segs:
            usr_segs[element[1]['Segment_Index']] = {'Arousal': arousal_anot.tolist(), 'Valence': valence_anot.tolist()}
    json.dump(data, open('../amigos.json', 'w'), indent=4)
