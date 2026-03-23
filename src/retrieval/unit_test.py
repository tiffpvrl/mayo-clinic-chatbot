from rag import retrieve_for_query

result = retrieve_for_query("can I drive after my colonoscopy?", patient_id='P658448776', is_follow_up=False)

print("=== Clinical Hits ===")
for h in result.clinical_hits:
    print(h['distance'], h['id'])
print(result.clinical_context)

print("\n=== Q&A Hits ===")
for h in result.qa_hits:
    print(h['distance'], h['id'])
print(result.qa_context)

print("\n=== Conversation Hits ===")
for h in result.conversation_hits:
    print(h['distance'], h['id'])
print(result.conversation_context)
