# ai_proto
Playing with tools w/ OpenAI) and w/ Langchain + Groq + Llama 3 70B.
Accessing my DB.  AI must plan multiple functions calls (chain of reasoning),
dependent on questions asked.  Here is a question:

```
{
    "question": "What is the PCI and number of people in these cities in California: Burlingame and  Palo Alto?"
}
```

Here is a *correct* answer from Llama 3 70B:

```
{
    "answer": "The Per Capita Income (PCI) and population of Burlingame, California are: $90,326 and 30,995, respectively. \n\nThe Per Capita Income (PCI) and population of Palo Alto, California are: $117,476 and 67,901, respectively."
}
```

Happy to fire this up if anyone want to test it.

# For requirements.txt
pipreqs . --force
