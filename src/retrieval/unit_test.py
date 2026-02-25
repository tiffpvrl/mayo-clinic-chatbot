from rag import retrieve_for_query


hits, context = retrieve_for_query("can I drive after my colonoscopy?")
for h in hits:
    print(h['distance'], h['id'])
print(context)