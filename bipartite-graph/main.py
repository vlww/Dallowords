import pandas as pd
import string
import re
import nltk
import networkx as nx
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# dont even need this anymore lol
# nltk.download('stopwords') 
stop_words = set(stopwords.words('english'))


df = pd.read_csv('dally.csv', header=None, names=['paragraph'])

# character list w alternate names
character_aliases = {
    'Clarissa Dalloway': ['clarissa', 'mrs dalloway', 'clarissas'],
    'Richard Dalloway': ['richard', 'richard dalloway', 'richards'],
    'Hugh Whitbread': ['hugh', 'hugh whitbread', 'whitbread', 'hughs'],
    'Peter Walsh': ['peter', 'peter walsh', 'walsh', 'peters'],
    'Lady Bruton': ['lady bruton', 'bruton'],
    'Sally Seton': ['sally', 'sally seton', 'seton', 'sallys', 'rosseter', 'lady rosseter'],
    'Elizabeth Dalloway': ['elizabeth', 'miss dalloway', 'elizabeths'],
    'Doris Kilman': ['kilman', 'miss kilman', 'doris'],
    'Septimus Warren Smith': ['septimus', 'mr warren smith'],
    'Lucrezia Warren Smith': ['rezia', 'lucrezia', 'mrs warren smith'],
    'Dr. Holmes': ['dr holmes', 'holmes', 'rezias'],
    'Sir William Bradshaw': ['bradshaw', 'sir william bradshaw', 'william', 'bradshaws']
}

# word list
words_to_check = ["old", "people", "felt", "man", "always", "without", "rather", "course", "day", "party", "lunch", "grey"]

# clean
def clean_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation.replace('.', '')))

cleaned_paragraphs = df['paragraph'].dropna().apply(clean_text)

# store word freq
word_occurrences = {character: {word: 0 for word in words_to_check} for character in character_aliases.keys()}
character_sentence_counts = {}

# character-word analysis
for character, names in character_aliases.items():
    
    sentences_with_character = set()  # avoid duplicate sentences

    # regex pattern for naming
    name_pattern = r'\b(?:' + '|'.join(map(re.escape, names)) + r')\b'
    for paragraph in cleaned_paragraphs:
        sentences = paragraph.split('.')  
        for sentence in sentences:
            if re.search(name_pattern, sentence):
                sentences_with_character.add(sentence.strip())

    total_sentences = len(sentences_with_character)
    character_sentence_counts[character] = total_sentences

    # count words
    for sentence in sentences_with_character:
        for word in words_to_check:
            if re.search(r'\b' + re.escape(word) + r'\b', sentence):
                word_occurrences[character][word] += 1

#print data to console
for character, word_counts in word_occurrences.items():
    total_sentences = character_sentence_counts.get(character, 1)
    print(f"{character} (total sentences: {total_sentences}):")

    for word, count in word_counts.items():
        percentage = (count / total_sentences) * 100 if total_sentences > 0 else 0
        print(f"  {word}: {count} ({percentage:.2f}%)")

    print()

df_matrix = pd.DataFrame(word_occurrences).T.fillna(0)
df_matrix = pd.DataFrame(word_occurrences).T.fillna(0).astype(float)

for character in df_matrix.index:
    total_sentences = character_sentence_counts.get(character, 1)
    df_matrix.loc[character] = (df_matrix.loc[character] / total_sentences) * 100  # percentage conversion

# make graph

G = nx.Graph()

characters = list(df_matrix.index)
words = list(df_matrix.columns)

for c in characters:
    G.add_node(c, type='character')
for w in words:
    G.add_node(w, type='word')

threshold = 1

for c in characters:
    for w in words:
        weight = df_matrix.loc[c, w]
        if weight >= threshold:
            G.add_edge(c, w, weight=weight)

pos = nx.spring_layout(G, weight='weight', k=0.5, iterations=200, seed=42)

plt.figure(figsize=(12, 8))

edges = G.edges(data=True)
edge_weights = [edata['weight'] * 0.3 for (_, _, edata) in edges]
nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='gray', alpha=0.5)

character_nodes = [node for node in G.nodes if G.nodes[node]["type"] == "character"]
word_nodes = [node for node in G.nodes if G.nodes[node]["type"] == "word"]

nx.draw_networkx_nodes(G, pos, nodelist=character_nodes, node_color="lightblue", node_shape="s", node_size=900)
nx.draw_networkx_nodes(G, pos, nodelist=word_nodes, node_color="orange", node_shape="o", node_size=600)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

plt.title("Force-Directed Bipartite Graph of Character-Word Connections", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()
