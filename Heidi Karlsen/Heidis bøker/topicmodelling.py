
from collections import Counter
import fork_kombo as fk


stop_words = Counter({'.': 895447698, ',': 755708767, 'og': 368375214, 'i': 314127096, 'det': 186406397, 'som': 179711677, 'til': 172646192, 'er': 170110193, 'av': 161509708, 'en': 157857924, 'på': 141617274, 'for': 137549491, 'at': 134427056, '-': 132523227, 'å': 127003302, 'med': 123295398, ')': 118073784, 'var': 104025227, '(': 102168357, 'den': 101656758, 'ikke': 93827828, 'de': 93487034, ':': 84550375, '«': 79059904, '»': 78408586, 'han': 77524313, 'har': 72230924, 'om': 71929211, 'et': 69332753, 'Det': 57008565, '?': 56509276, 'seg': 55436529, 'jeg': 53176919, 'kan': 51107817, 'hadde': 48348683, '\u2014': 48247274, 'fra': 46609687, 'the': 45795771, 'I': 45437940, 'så': 43907451, ';': 43740798, '"': 41371737, 'eller': 39877232, '/': 39695412, 'vi': 38041730, 'a': 36777496, "'": 36680804, 'men': 36508750, '1': 36408861, 'ved': 36250443, '2': 35592673, 'du': 34358343, 'hun': 33308986, 'of': 31515341, '!': 31092367, '3': 30668952, 'to': 30392321, '^': 30278803, 'ble': 29027839, 'der': 28819304, 'Han': 27426261, 'Jeg': 27059347, 'skal': 26652041, 'vil': 26356007, 'ut': 26207653, 'over': 25444437, 'også': 24884067, 'være': 24774392, 'da': 23619171, 'and': 23605246, 'De': 22618787, 'etter': 22606167, 'Men': 22320644, 'dette': 22110102, '*': 21946002, 'in': 21759025, 'ham': 21049765, 'meg': 20385977, 'kunne': 20325553, '6': 20281748, 'andre': 20140440, 'opp': 20023927, 'sa': 19861172, 's': 19417663, 'denne': 19310450, 'sin': 18561013, 'noe': 18147286, 'alle': 17998281, 'dem': 17724702, 'inn': 16948929, 'Den': 16886056, '4': 16806594, 'blir': 16797262, 'må': 16792527, 'f': 16778588, 'mot': 16763924, 'bare': 16685232, 'man': 16511054, 'ha': 15869213, 'kom': 15810618, 'enn': 15444323, 'Og': 15128313, '5': 14979648, 'hans': 14858092, 'når': 14687279, 'noen': 14356466, 'ville': 14332841, 'skulle': 14032370, 'under': 13661777, '=': 13645666, 'få': 13514358, 'Hun': 13483002, 'En': 12890838, 'se': 12841348, 'her': 12829240, 'slik': 12815263, '>': 12735640, 'm': 12715825, 'af': 12682434, 'år': 12652095, 'henne': 12314568, 'mellom': 12312448, 'hvor': 12284620, 'mer': 12140654, 'Vi': 12099755, 'mange': 11659336, 'bli': 11397443, 'o': 11367100, 'paa': 11140979, 'før': 10930201, 'nå': 10775087, 'ein': 10696573, 'deg': 10595745, 'vært': 10499485, 'oss': 10198453, 'igjen': 10103486, 'samme': 10026427, 'disse': 9951377, 'dei': 9947527, 'alt': 9763664, '8': 9614544, 'sine': 9491013, '7': 9401072, 'gikk': 9319658, 'ned': 9050315, 'første': 9033018, 'fikk': 8954637, 'hva': 8838342, '§': 8748480, '\u2022': 8728722, 'hele': 8725361, 'store': 8578569, '10': 8568583, 'sammen': 8430370, 'kommer': 8387288, 'Dette': 8370795, '0': 8365080, 'e': 8317949, 'ingen': 8297237, 'is': 8228342, 'sig': 8173271, 'ta': 8073607, 'n': 8044371, 'gjennom': 7997686, 'tid': 7982223, 'For': 7841384, '<': 7766646, 'uten': 7570085, 'får': 7546989, 'gang': 7522691, '%': 7504135, 'stor': 7443305, 'la': 7432484, 'går': 7292729, ']': 6934526, 'gjøre': 6599257, 'r': 6560310, 'The': 6265404, '\u25a0': 5927098, '9': 5926487, 'Du': 5793631, '+': 5539077, '[': 5426293, 'nok': 5418705, 'Da': 5387451, 't': 5196728, 'både': 5176555, 'ser': 5128852, 'seiv': 5102612, 'selv': 5102135, 'min': 5100944, 'måtte': 5092817, 'godt': 5073937, 'flere': 5057679, 'saa': 5006526, 'sitt': 4610802, 'tok': 4507691, 'dag': 4455387, 'A': 4372431, 'si': 4335253, 'c': 4302728, 'gå': 4288226, 'helt': 4107967, 'b': 3940654, 'x': 3922276, 'pa': 3855255, 'litt': 3808802, 'blitt': 3713728, 'be': 3689844, 'del': 3624860, 'blev': 3597170, 'mig': 3565177, 'l': 3481747, 'ikkje': 3459106, '©': 3350602, 'that': 3157188, 'vel': 3116364, 'siden': 3107511, '...': 3083317, 'efter': 2972153, 'havde': 2888681, 'meget': 2862796, 'ei': 2858809, 'tilbake': 2809669, '~': 2754750, 'with': 2697444, 'nu': 2686707, 'on': 2673957, 'd': 2669631, 'as': 2654885, 'Så': 2616521, 'by': 2608468, 'kunde': 2599501, 'hennes': 2508802, '\\': 2459932, 'annen': 2438341, 'are': 2414728, 'mens': 2326296, '12': 2322507, 'mye': 2257462, 'annet': 2230368, 'skulde': 2202168, 'rundt': 2162195, 'O': 2157049, 'hvordan': 2150009, 'Hva': 2122944, 'Bom': 2090216, 'vilde': 2056172, 'vet': 2034595, 'ogsaa': 1993849, 'jo': 1802040, 'end': 1737395, 'op': 1682960, '\u201e': 1681498, 'maa': 1612373, 'have': 1593013, 'hvad': 1542238, 'gamle': 1404487, 'nar': 1376468, 'os': 1354091, 'dog': 1334173, 'faa': 1322198, 'noget': 1214573, 'fram': 1208470, 'deres': 1192429, 'På': 1187589, '\x84': 1178708, 'dig': 1134106, 'mere': 1128633, 'hvis': 1125545, 'sit': 1125177, 'kun': 1117542, 'eg': 1114877, 'hos': 1109619, 'komme': 1101523, 'kor': 1087456, 'nogen': 1052857, 'mod': 1036004, 'so': 958431, 'ind': 957796, '£': 937864, 'naar': 919992, 'mellem': 913501, 'nan': 905159, 'Der': 886128, 'ud': 869142, 'anden': 858507, 'ere': 836740, 'maatte': 830392, 'Gud': 814132, 'ben': 812002, 'hende': 765265, 'die': 756106, 'uden': 736002, 'gik': 715517, 'bet': 704583, 'mann': 649241, 'blive': 645044, 'Når': 623193, 'nye': 618348, 'stod': 617093, 'sagde': 611853, 'mit': 608841, 'saaledes': 599747, 'været': 562968, 'F': 559037, 'Ja': 556513, 'an': 550507, 'endnu': 550284, 'S.': 540004, 'forn': 502066, 'aa': 489950, 'Tid': 486268, 'stal': 483600, 'frem': 475492, 'al': 463573, '$': 453163, 'ber': 425026, 'B': 423023, 'k': 409635, '11': 407750, 'lor': 402199, 'fik': 398567, 'meb': 396114, 'sier': 395778, 'fom': 391641, 'und': 390706, 'lian': 390143, 'derfor': 374560, 'gaa': 363273, 'aldri': 352521, 'thi': 350277, '03': 336908, 'gjorde': 330696, 'sto': 296657, 'dere': 292098, 'rett': 289940, 'kanskje': 287689, 'vare': 285188, 'gjør': 284181, 'bort': 280203, 'sett': 279656, 'och': 276378, 'satt': 273861, 'tre': 267408, 'visste': 267198, 'like': 266554, 'vor': 239897, 'Aar': 234030, '&': 224849, 'hendes': 224226, 'nåar': 221286, 'Guds': 202060, 'vcere': 197851, 'v.': 186558, 'bliver': 177544, 'iffe': 171710, 'Ord': 165862, 'aar': 162089, 'ok': 149812, 'tit': 140334, 'me': 134678, 'udi': 134326, '@': 133756, 'fig': 133446, 'v': 132875, 'ho': 130980, 'tilbage': 128871, 'des': 122999, 'ja': 122198, 'II': 118166, 'gaar': 113582, 'zu': 105445, 'das': 94995, '|': 88274, 'din': 87327, 'S': 62507, 'V.': 57916, 'von': 53150, 'I.': 52894, 'ang': 48383, '°': 45381, 'el': 40480, 'was': 39582, 'samt': 39321, 'X': 37304, 'it': 36522, 'sich': 35918, 'which': 35664, 'u': 29559, '23': 29502, 'imod': 29063, 'Mand': 27881, 'ist': 27723, 'hvilken': 27080, 'his': 26455, 'or': 26042, 'es': 25986, 'saadan': 25858, 'cl': 25372, 'ligesom': 25232, 'see': 24899, 'ogs': 24759, 'ich': 23794, 'nicht': 23131, 'eine': 19556, 'auf': 16143, 'im': 15850, 'from': 15784, 'als': 14949, 'auch': 14735, 'ar': 14728, 'bie': 14206, 'plur': 13407, 'unb': 9751, 'le': 8955, 'sie': 8567, 'my': 8393, 'les': 8385, 'Ich': 8173, 'Jeronimus': 8121, 'Pl': 7961, 'mir': 7794, 'had': 7746, 'g': 7688, 'not': 7683, 'oder': 7580, 'aber': 7528, 'they': 7509, 'j': 7442, 'he': 7305, 'Z': 7249, 'ihr': 7067, 'te': 6787, 'mich': 6759, 'Ach': 6756, 'Pernille': 6743, 'Henrich': 6721, 'wird': 6503, 'z': 6470, 'but': 6439, '25': 6336, 'Herr': 6312, 'H': 6123, 'their': 6102, 'this': 5996})

def corpusfreq(my_corpus):
    
    freqs = Counter()
    for text in my_corpus:
        freqs.update(text)
    return freqs

def files2corpus(filelist):
    corpus = []
    for file in filelist:
        corpus.append(xml2text(file))
    return corpus

def countertop(col, top=100):
    """Fetch the topmost elements from a counter as a counter"""
    c = Counter()
    for (x,y) in col.most_common(100):
        c[x] = y
    return c

def find_offset(text_tokens, word_tokens, offset, num_of_result):
    y = 0
    x = 0
    num_of_loops = 0
    #x is a control variable that holds the exit status. It is set at various places.
    while x != -1 and num_of_loops <= offset*num_of_result:
        try:
            indices = [text_tokens[y:].index(w) for w in word_tokens]
        except:
            x = -1

        if x != -1:
            if indices != []:
                x = min(indices)
            else:
                x = -1
        y = y + x + 1
        num_of_loops += 1
    return y

def conc(text_tokens, search_words, spaced=10, window=5, num_of_result=10, offset=0):
    word_tokens = search_words.split()
    #print(word_tokens)
    
    y = find_offset(text_tokens, word_tokens, offset, num_of_result)
    #print(y)
    x = 0
    result_conc = []
    num_of_loops = 0
    #x is a control variable that holds the exit status. It is set at various places.
    while x != -1 and num_of_loops <= num_of_result:
        try:
            indices = [text_tokens[y:].index(w) for w in word_tokens]
            #print(indices)
        except:
            x = -1

        if x != -1:
            if indices != []:
                if max(indices) - min(indices) <= spaced:
                    #print(max(indices), min(indices))
                    result_conc +=[' '.join(text_tokens[max(0, y - window) : min(y + window, len(text_tokens))])]
                    #print(result_conc)
                x = min(indices)
            else:
                x = -1
        y = y + x + 1
        num_of_loops += 1
    return result_conc

def chunker(text_to_chunk, chunk_num =20, stop_words = dict()):
    tokens = text_to_chunk
    if len(stop_words.keys()) > 0: 
        tokens = [token for token in tokens if token not in stop_words.keys()]
    chunks = int(len(tokens)/chunk_num)
    #print(chunks)
    chunk_text = []
    for i in range(chunk_num):
        chunk_text += [Counter(tokens[i * chunks:(i + 1) * chunks])]
    return chunk_text

def chunk_tokens(tokens, chunk_size =200, stop_words = {',':1, '.':2}):
    if len(stop_words.keys()) > 0: 
        tokens = [token for token in tokens if token not in stop_words.keys()]
    chunks = min(chunk_size, len(tokens))
    chunk_num = int(len(tokens)/chunk_size) + 1
    chunk_text = []
    for i in range(chunk_num):
        chunk_text += [Counter(tokens[i * chunks:(i + 1) * chunks])]
    return chunk_text

def xml2text(filepath, nodename='*'):
    from lxml import etree

    tree = etree.parse(filepath)
    root = tree.getroot()
    text = ""
    if nodename != '*':
        for node in root.findall('.//'):
            tag = etree.QName(node)
            if tag.localname == nodename:
                for x in node.findall('.//'):
                    if x.text != None:
                        text += x.text  
    else:
        for node in root.findall('.//'):
            tag = etree.QName(node)
            for x in node.findall('.//'):
                if x.text != None:
                    text += x.text  
    return fk.tokenize(text)

def html2text(in_text, path="body//*"):
    import xml.etree.ElementTree as ET

    tree = ET.parse(in_text)
    root = tree.getroot()
    text = ""
    for x in root.findall(path):
        if x.text != None:
            text += x.text + " "
    return text

def html2text_div(in_text):
    import xml.etree.ElementTree as ET

    tree = ET.parse(in_text)
    root = tree.getroot()
    text = ""
    for x in root.findall(".//*[@class='para']"):
        if x.text != None:
            text += x.text + " "
    return text

def corpus_files(corpus_path):
    import os
    (_ ,_ , corpus_files) =  next(os.walk(corpus_path))
    return [os.path.join(corpus_path, corpus_f) for corpus_f in corpus_files]


def json2dict4urn(jsonstring, order = 'assoc'):
    import json

    struct = json.loads(jsonstring.text)
    result = dict()
    for i in range(len(struct)):
        result[struct[i]['text']] = struct[i][order]
    return result


def nb_topics(chunks, unigram):
    
    import numpy as np
    import sqlite3
    from collections import Counter
    
    with sqlite3.connect(unigram) as con:
        cur = con.cursor()
        totals = 10000000000
        chunks_assoc = []
        for c in chunks:
            d = dict()
            cs = sum(c.values())
            for w in c:
                query = cur.execute('select freq from unigram where first = ?', (w,)).fetchall() 
                if query != []:
                    freq_global = query[0][0]
                else:
                    freq_global = 0
                val = 0
                if  freq_global > 0:
                    val =  np.sqrt(c[w])*np.log(c[w]*totals/(cs*freq_global))/np.log(2)
                if val > 0:
                    d[w] = val
            chunks_assoc += [Counter(d)]
    return chunks_assoc

def lookup(w):
    import json
    import requests as rq
    
    uniurl = ("http://www.nb.no/sp_tjenester/beta/ngram_1/freq/query?terms=%s" % (w,))
    r = json.loads(rq.get(uniurl).text)
    if r[0]['result'] != []:
        res = r[0]['result'][0]['sum(freq)']
    else:
        res = 0
    return res


def get_stop_words():
    
    import json
    from collections import Counter
    import requests as rq
    
    uniurl = ("http://www.nb.no/sp_tjenester/beta/ngram_1/freq/query?terms=*")
    return Counter(json2dict4uni(rq.get(uniurl)))

def show_topics(topic_list, number_of_words= 10):
    for i in range(len(topic_list)):
        words = ", ".join([f[0] for f in topic_list[i].most_common(number_of_words)])
        print("Topic %s: %s" % (i,words))    

def json2dict4uni(jsonstring):
 
    import json
    struct = json.loads(jsonstring.text)
    struct = struct[0]['result']
    result = dict()
    for i in range(len(struct)):
        result[struct[i]['first']] = struct[i]['sum(freq)']
    return result

def urn2dict(urn):
    from collections import Counter
    import requests as rq
    #import urllib
   
    urn_url_freq = "http://www.nb.no/sp_tjenester/beta/ngram_1/assoc/query?urn=%s"
    urn_words = rq.get(urn_url_freq % urn)
    return Counter(json2dict4urn(urn_words, order='pmi'))

def make_matrix(chunks):
    from sklearn.feature_extraction import DictVectorizer 
    v = DictVectorizer(sparse=True)
    D = chunks
    X = v.fit_transform(D)
    return X

def topic_model_nmf1(num_of_topics = 20, max_iterations=200):
    decomposition.NMF(init="nndsvd", n_components=num_of_topics, max_iter=max_iterations)
    W = model.fit_transform(X)
    H = model.components_
    return H

def idf(chunks):
    for p in chunks:
        for w in p:
            df = len([chunk for chunk in chunks if w in chunk])
            p[w] = p[w]*1/df

def topic_model_nmf(chunks, num_of_topics=20, topic_size=10, max_iterations=200):
    
    from sklearn.feature_extraction import DictVectorizer 
    from sklearn import decomposition
    import numpy as np
    
    v = DictVectorizer(sparse=True)
    D = chunks
    X = v.fit_transform(D)

    num_terms = len(v.vocabulary_)
    terms = [""] * num_terms
    for term in v.vocabulary_.keys():
        terms[ v.vocabulary_[term] ] = term

    model = decomposition.NMF(init="nndsvd", n_components=num_of_topics, max_iter=max_iterations)
    W = model.fit_transform(X)
    H = model.components_
    topics = []
    for topic_index in range( H.shape[0] ):
        top_indices = np.argsort( H[topic_index,:] )[::-1][0:topic_size]
        term_ranking = [terms[i] for i in top_indices]
        #print(term_ranking)
        topics += [ ", ".join( term_ranking )] 
    return topics
        
        
def modeller_set(list_of_texts, tema=10, chunk_size=1000, topic_size=20, stop_words=stop_words):
    nmf_chunks = []
    for chunks in map(lambda tokens: chunk_tokens(tokens, chunk_size, stop_words), list_of_texts):
        nmf_chunks += chunks
    return topic_model_nmf(nmf_chunks, num_of_topics = tema, topic_size = topic_size)       

def modeller(tokens, tema=10, chunk_size=1000, topic_size=20, stop_words=stop_words):
    nmf_chunks = chunk_tokens(tokens, chunk_size, stop_words)
    return topic_model_nmf(nmf_chunks, num_of_topics = tema, topic_size = topic_size)