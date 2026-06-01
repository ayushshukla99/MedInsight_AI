from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key="AIzaSyBha8Cv7H68STfT5ucE1NMzj-jgkKsI--s"
)

response = llm.invoke("how many footballers from portugal play in premier league")

print(response.text)