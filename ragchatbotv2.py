import os
import time
from pinecone import Pinecone, ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Set your API keys
os.environ['PINECONE_API_KEY'] = ''
os.environ['OPENAI_API_KEY'] = ''

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Define embeddings
model_name = 'text-embedding-3-small'
embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)

# Define index and namespace
index_name = "wedding-music-bot"
namespace = "weddingvector"

# Setup Pinecone index
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=spec
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Prepare document for search

markdown_document = """
## Wedding Music Recommendation Dataset

This dataset is designed to provide a selection of songs suitable for weddings, encompassing various genres and moments within the event, such as the ceremony, reception, and first dance.

### Music Entries

## Perfect
- **Artist**: Ed Sheeran
- **Genre**: Pop
- **Year**: 2017
- **Description**: A romantic ballad ideal for the first dance.
- **Popularity**: 92
- **Duration (ms)**: 263400
- **Explicit**: No
- **Album**: Divide

## All of Me
- **Artist**: John Legend
- **Genre**: R&B
- **Year**: 2013
- **Description**: A soulful love song perfect for the first dance.
- **Popularity**: 95
- **Duration (ms)**: 269000
- **Explicit**: No
- **Album**: Love in the Future

## A Thousand Years
- **Artist**: Christina Perri
- **Genre**: Pop
- **Year**: 2011
- **Description**: A beautiful ballad often used during the ceremony.
- **Popularity**: 90
- **Duration (ms)**: 285400
- **Explicit**: No
- **Album**: The Twilight Saga: Breaking Dawn – Part 1 (Original Motion Picture Soundtrack)

## Marry Me
- **Artist**: Train
- **Genre**: Pop Rock
- **Year**: 2009
- **Description**: A romantic song perfect for the proposal or ceremony.
- **Popularity**: 88
- **Duration (ms)**: 221640
- **Explicit**: No
- **Album**: Save Me, San Francisco

## Can't Help Falling in Love
- **Artist**: Elvis Presley
- **Genre**: Rock
- **Year**: 1961
- **Description**: A classic love song perfect for any wedding moment.
- **Popularity**: 94
- **Duration (ms)**: 178000
- **Explicit**: No
- **Album**: Blue Hawaii

## I Don't Want to Miss a Thing
- **Artist**: Aerosmith
- **Genre**: Rock
- **Year**: 1998
- **Description**: A powerful rock ballad ideal for the first dance.
- **Popularity**: 89
- **Duration (ms)**: 300300
- **Explicit**: No
- **Album**: Armageddon: The Album

## Thinking Out Loud
- **Artist**: Ed Sheeran
- **Genre**: Pop
- **Year**: 2014
- **Description**: A romantic ballad perfect for the first dance.
- **Popularity**: 93
- **Duration (ms)**: 281400
- **Explicit**: No
- **Album**: x

## You Are the Best Thing
- **Artist**: Ray LaMontagne
- **Genre**: Soul
- **Year**: 2008
- **Description**: A soulful song ideal for the reception.
- **Popularity**: 85
- **Duration (ms)**: 221600
- **Explicit**: No
- **Album**: Gossip in the Grain

## At Last
- **Artist**: Etta James
- **Genre**: Blues
- **Year**: 1960
- **Description**: A timeless classic perfect for the first dance.
- **Popularity**: 91
- **Duration (ms)**: 182600
- **Explicit**: No
- **Album**: At Last!

## Make You Feel My Love
- **Artist**: Adele
- **Genre**: Pop
- **Year**: 2008
- **Description**: A heartfelt ballad suitable for any wedding moment.
- **Popularity**: 88
- **Duration (ms)**: 215200
- **Explicit**: No
- **Album**: 19

## From This Moment On
- **Artist**: Shania Twain
- **Genre**: Country
- **Year**: 1997
- **Description**: A romantic country ballad ideal for the ceremony or first dance.
- **Popularity**: 87
- **Duration (ms)**: 229000
- **Explicit**: No
- **Album**: Come On Over

## Better Together
- **Artist**: Jack Johnson
- **Genre**: Folk
- **Year**: 2005
- **Description**: A laid-back love song perfect for the reception.
- **Popularity**: 84
- **Duration (ms)**: 207600
- **Explicit**: No
- **Album**: In Between Dreams

## Here Comes the Sun
- **Artist**: The Beatles
- **Genre**: Rock
- **Year**: 1969
- **Description**: A cheerful and uplifting song ideal for the ceremony.
- **Popularity**: 89
- **Duration (ms)**: 185000
- **Explicit**: No
- **Album**: Abbey Road

## The Way You Look Tonight
- **Artist**: Frank Sinatra
- **Genre**: Jazz
- **Year**: 1964
- **Description**: A classic love song perfect for the first dance.
- **Popularity**: 90
- **Duration (ms)**: 210000
- **Explicit**: No
- **Album**: Sinatra Sings Days of Wine and Roses, Moon River, and Other Academy Award Winners

## My Girl
- **Artist**: The Temptations
- **Genre**: R&B
- **Year**: 1965
- **Description**: A feel-good classic ideal for the reception.
- **Popularity**: 88
- **Duration (ms)**: 166000
- **Explicit**: No
- **Album**: The Temptations Sing Smokey

## Endless Love
- **Artist**: Diana Ross & Lionel Richie
- **Genre**: R&B
- **Year**: 1981
- **Description**: A duet perfect for the first dance or ceremony.
- **Popularity**: 91
- **Duration (ms)**: 262000
- **Explicit**: No
- **Album**: Endless Love Soundtrack

## Isn't She Lovely
- **Artist**: Stevie Wonder
- **Genre**: R&B
- **Year**: 1976
- **Description**: A joyful song perfect for celebrating the bride.
- **Popularity**: 87
- **Duration (ms)**: 376200
- **Explicit**: No
- **Album**: Songs in the Key of Life

## Marry You
- **Artist**: Bruno Mars
- **Genre**: Pop
- **Year**: 2010
- **Description**: A fun and upbeat song ideal for the reception.
- **Popularity**: 89
- **Duration (ms)**: 225800
- **Explicit**: No
- **Album**: Doo-Wops & Hooligans

## L-O-V-E
- **Artist**: Nat King Cole
- **Genre**: Jazz
- **Year**: 1965
- **Description**: A timeless jazz classic perfect for any wedding moment.
- **Popularity**: 85
- **Duration (ms)**: 146000
- **Explicit**: No
- **Album**: L-O-V-E

## Everything
- **Artist**: Michael Bublé
- **Genre**: Jazz
- **Year**: 2007
- **Description**: A jazzy love song ideal for the first dance or reception.
- **Popularity**: 86
- **Duration (ms)**: 230000
- **Explicit**: No
- **Album**: Call Me Irresponsible

## You Make My Dreams
- **Artist**: Daryl Hall & John Oates
- **Genre**: Pop Rock
- **Year**: 1980
- **Description**: An upbeat, feel-good song perfect for the reception.
- **Popularity**: 87
- **Duration (ms)**: 188000
- **Explicit**: No
- **Album**: Voices

## Ho Hey
- **Artist**: The Lumineers
- **Genre**: Indie Folk
- **Year**: 2012
- **Description**: A catchy indie folk song ideal for the reception.
- **Popularity**: 88
- **Duration (ms)**: 163000
- **Explicit**: No
- **Album**: The Lumineers

## I Will Always Love You
- **Artist**: Whitney Houston
- **Genre**: R&B
- **Year**: 1992
- **Description**: A powerful ballad perfect for the first dance.
- **Popularity**: 92
- **Duration (ms)**: 273000
- **Explicit**: No
- **Album**: The Bodyguard: Original Soundtrack Album

## I Choose You
- **Artist**: Sara Bareilles
- **Genre**: Pop
- **Year**: 2013
- **Description**: A heartfelt song ideal for the ceremony or first dance.
- **Popularity**: 85
- **Duration (ms)**: 263800
- **Explicit**: No
- **Album**: The Blessed Unrest

## Happy
- **Artist**: Pharrell Williams
- **Genre**: Pop
- **Year**: 2013
- **Description**: An upbeat and joyful song perfect for the reception.
- **Popularity**: 90
- **Duration (ms)**: 233000
- **Explicit**: No
- **Album**: G I R L

## Home
- **Artist**: Edward Sharpe & The Magnetic Zeros
- **Genre**: Indie Folk
- **Year**: 2010
- **Description**: A whimsical indie folk song ideal for the reception.
- **Popularity**: 87
- **Duration (ms)**: 325000
- **Explicit**: No
- **Album**: Up from Below

## Your Song
- **Artist**: Elton John
- **Genre**: Pop
- **Year**: 1970
- **Description**: A classic love song perfect for any wedding moment.
- **Popularity**: 88
- **Duration (ms)**: 241000
- **Explicit**: No
- **Album**: Elton John

## Just the Way You Are
- **Artist**: Bruno Mars
- **Genre**: Pop
- **Year**: 2010
- **Description**: A sweet and romantic song perfect for the first dance.
- **Popularity**: 91
- **Duration (ms)**: 220000
- **Explicit**: No
- **Album**: Doo-Wops & Hooligans

## Wonderful Tonight
- **Artist**: Eric Clapton
- **Genre**: Rock
- **Year**: 1977
- **Description**: A romantic rock ballad ideal for the first dance.
- **Popularity**: 89
- **Duration (ms)**: 216000
- **Explicit**: No
- **Album**: Slowhand

## I’m Yours
- **Artist**: Jason Mraz
- **Genre**: Pop
- **Year**: 2008
- **Description**: A feel-good, laid-back song ideal for the reception.
- **Popularity**: 89
- **Duration (ms)**: 242000
- **Explicit**: No
- **Album**: We Sing. We Dance. We Steal Things.

## You and Me
- **Artist**: Lifehouse
- **Genre**: Rock
- **Year**: 2005
- **Description**: A romantic rock song perfect for the first dance.
- **Popularity**: 85
- **Duration (ms)**: 235000
- **Explicit**: No
- **Album**: Lifehouse

## You Are the Love of My Life
- **Artist**: Sam Cooke
- **Genre**: Soul
- **Year**: 1964
- **Description**: A classic soul song ideal for the ceremony.
- **Popularity**: 85
- **Duration (ms)**: 200000
- **Explicit**: No
- **Album**: Ain't That Good News

## I Will Follow You into the Dark
- **Artist**: Death Cab for Cutie
- **Genre**: Indie Rock
- **Year**: 2005
- **Description**: A touching indie rock ballad suitable for the ceremony.
- **Popularity**: 86
- **Duration (ms)**: 215000
- **Explicit**: No
- **Album**: Plans

## Love Me Like You Do
- **Artist**: Ellie Goulding
- **Genre**: Pop
- **Year**: 2015
- **Description**: A pop love song ideal for the first dance or reception.
- **Popularity**: 89
- **Duration (ms)**: 251000
- **Explicit**: No
- **Album**: Fifty Shades of Grey (Original Motion Picture Soundtrack)

## Halo
- **Artist**: Beyoncé
- **Genre**: Pop
- **Year**: 2008
- **Description**: A powerful love song perfect for the first dance.
- **Popularity**: 92
- **Duration (ms)**: 261000
- **Explicit**: No
- **Album**: I Am... Sasha Fierce

## Marry Me
- **Artist**: Jason Derulo
- **Genre**: Pop
- **Year**: 2013
- **Description**: A romantic pop song perfect for the proposal or first dance.
- **Popularity**: 86
- **Duration (ms)**: 218000
- **Explicit**: No
- **Album**: Tattoos

## Speechless
- **Artist**: Dan + Shay
- **Genre**: Country
- **Year**: 2018
- **Description**: A romantic country song ideal for the first dance.
- **Popularity**: 87
- **Duration (ms)**: 213000
- **Explicit**: No
- **Album**: Dan + Shay

## Die a Happy Man
- **Artist**: Thomas Rhett
- **Genre**: Country
- **Year**: 2015
- **Description**: A heartfelt country song perfect for the first dance.
- **Popularity**: 88
- **Duration (ms)**: 235000
- **Explicit**: No
- **Album**: Tangled Up

## XO
- **Artist**: Beyoncé
- **Genre**: Pop
- **Year**: 2013
- **Description**: A heartfelt love song ideal for the reception.
- **Popularity**: 85
- **Duration (ms)**: 209000
- **Explicit**: No
- **Album**: Beyoncé

## Best Day of My Life
- **Artist**: American Authors
- **Genre**: Indie Rock
- **Year**: 2014
- **Description**: An upbeat indie rock song perfect for the reception.
- **Popularity**: 88
- **Duration (ms)**: 205000
- **Explicit**: No
- **Album**: Oh, What a Life

## Latch (Acoustic)
- **Artist**: Sam Smith
- **Genre**: Pop
- **Year**: 2013
- **Description**: A stripped-down, acoustic love song ideal for the first dance.
- **Popularity**: 86
- **Duration (ms)**: 187000
- **Explicit**: No
- **Album**: In the Lonely Hour (Drowning Shadows Edition)

## To Make You Feel My Love
- **Artist**: Bob Dylan
- **Genre**: Folk
- **Year**: 1997
- **Description**: A classic folk love song perfect for the ceremony or first dance.
- **Popularity**: 84
- **Duration (ms)**: 200000
- **Explicit**: No
- **Album**: Time Out of Mind

## Lucky
- **Artist**: Jason Mraz & Colbie Caillat
- **Genre**: Pop
- **Year**: 2008
- **Description**: A duet ideal for the first dance or reception.
- **Popularity**: 85
- **Duration (ms)**: 202000
- **Explicit**: No
- **Album**: We Sing. We Dance. We Steal Things.

## You’re Still the One
- **Artist**: Shania Twain
- **Genre**: Country
- **Year**: 1997
- **Description**: A romantic country ballad perfect for the first dance.
- **Popularity**: 87
- **Duration (ms)**: 202000
- **Explicit**: No
- **Album**: Come On Over

## Conclusion

This dataset provides a comprehensive collection of songs across various genres and years, perfect for generating music recommendations for weddings. Use this dataset to enhance the music recommendation capabilities of your chatbot.

"""


headers_to_split_on = [("##", "Header 2")]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_header_splits = markdown_splitter.split_text(markdown_document)

print(md_header_splits)

# Create vector store
docsearch = PineconeVectorStore.from_documents(
    documents=md_header_splits,
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace
)
time.sleep(1)

index = pc.Index(index_name)


for ids in index.list(namespace=namespace):
    query = index.query(
        id=ids[0],
        namespace=namespace,
        top_k=2,
        include_values=True,
        include_metadata=True
    )
    # print("first version query print", query)



# Initialize the LLM and QA system
llm = ChatOpenAI(
    openai_api_key=os.environ.get('OPENAI_API_KEY'),
    model_name='gpt-3.5-turbo',
    temperature=0.2
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)

def get_answers(query):
    # return qa.invoke(query)
    persona = """"
You are a professional wedding assistant with extensive knowledge of wedding music planning and organization. You have experience in assisting couples with every aspect of their wedding, especially choosing the right music for the ceremony and reception. You are detail-oriented, resourceful, and dedicated to making each wedding unique and memorable. 

When interacting with users, first, always ask about their preference, such as the style of the wedding(outdoor/indoor/small/large) and the taste of music such as the artists, year, the provide thoughtful and tailored advice based on their needs and preferences. Here are some examples of the types of assistance you can offer:

 **Music Recommendations**: Base on the provided dataset, suggest appropriate songs for different parts of the wedding, such as the ceremony, first dance, and reception. Offer a mix of classic and contemporary options that suit the couple's tastes.

If you give answer about music, not too vague, must enroll at least one of the specific piece of music in the dataset.


When responding to queries, be polite, empathetic, and professional. Your goal is to help users plan their perfect wedding day with ease and confidence.

If the user's query out of the scale of dataset, be polite to tell them who you are and what you are specialize in, and professional tell them you do not know because it's not your are specialize in, invite the user to ask questions about wedding music.
"""

    full_query = f"{persona} {query}"
    response = qa.invoke(full_query)
    return response['result']


# calculation part =====================


# =========以下是尝试print retrieve context
def retrieve_contexts(query):
    # Generate query embedding
    query_embedding = embeddings.embed_query(query)
    
    # Retrieve contexts from Pinecone
    results = index.query(
        vector=query_embedding,
        namespace=namespace,
        top_k=3,  # Adjust the number of retrieved contexts
        include_values=True,
        include_metadata=True
    )

    # print("results are ",results)

    # Extract retrieved contexts
    retrieved_contexts = [match['metadata']['text'] for match in results['matches']]


    # Print retrieved contexts
    for match in results['matches']:
        print(f"Score: {match['score']}")
        print(f"Context: {match['metadata']['text']}")
        print("\n")
        
    
    return retrieved_contexts
    

    # return results

# Example queries
query1 = "Can you recommend some songs for a beach wedding ceremony?"
query2 = "What are some classical music suitable for the first dance?"

# Retrieve and print contexts for each query
print("Query 1: Can you recommend some songs for a beach wedding ceremony")
contexts_query1 = retrieve_contexts(query1)
print("\nQuery 2: What are some classical music suitable for the first dance?")
contexts_query2 = retrieve_contexts(query2)

# ============以上是尝试print retrieve context

# Define relevant contexts for evaluation
relevant_contexts = {
    "Can you recommend some songs for a beach wedding ceremony?": [
        "Better Together",
        "Here Comes the Sun"
    ],
    "Give me some classical music suitable for the first dance": [
        "Canon in D",
        "Clair de Lune"
    ]
}

# Function to calculate precision
def calculate_precision(retrieved, relevant):
    relevant_retrieved = [context for context in retrieved if any(rel in context for rel in relevant)]
    precision = len(relevant_retrieved) / len(retrieved) if retrieved else 0
    return precision

# Example queries
queries = [
    "Can you recommend some songs for a beach wedding ceremony?",
    "Give me some classical music suitable for the first dance"
]

# Calculate and print precision for each query
for query in queries:
    print(f"Query: {query}")
    retrieved_contexts = retrieve_contexts(query)
    precision = calculate_precision(retrieved_contexts, relevant_contexts[query])
    print(f"Precision: {precision:.2f}")
    print("Retrieved Contexts:")
    for context in retrieved_contexts:
        print(f"- {context}")
    print("\n")

# Sample queries and comparation between chatbot with knowledge and without knowledge
# query1 = "What are your recommendation for wedding hold indoor?"

# query2 = "I like classical music, give me some advice"

# query3 = "What are the most popular song of spotify 2023?"

# query1_with_knowledge = qa.invoke(get_answers(query1))
# query1_without_knowledge = llm.invoke(query1)

# print(query1_with_knowledge)
# print()
# print(query1_without_knowledge)

# query2_with_knowledge = qa.invoke(get_answers(query2))
# query2_without_knowledge = llm.invoke(query2)

# print(query2_with_knowledge)
# print()
# print(query2_without_knowledge)

# query3_with_knowledge = qa.invoke(get_answers(query3))
# query3_without_knowledge = llm.invoke(query3)

# print(query3_with_knowledge)
# print()
# print(query3_without_knowledge)
