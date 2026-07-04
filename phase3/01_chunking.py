def chunk_document(text, chunk_size=200, overlap=50):
    """
    Breaks text into chunks of approximately chunk_size characters.
    overlap: how many characters to repeat between chunks
    so we dont lose context at chunk boundaries.
    """

    chunks = []
    start = 0

    while start < len(text):
        end=start+chunk_size
        chunk=text[start:end]
        chunks.append(chunk)
        start=end-overlap  # move start forward by chunk_size - overlap

    return chunks

def main():
    # Sample HR policy document
    document = """
    Sick leave entitlement is 12 days per year for all full time employees.
    Sick leave can be carried forward up to 5 days to the next calendar year.
    Employees must notify their manager before 9 AM on the day of absence.
    Medical certificate is required for sick leave exceeding 3 consecutive days.
    M
    aternity leave is 26 weeks as per company policy and government regulations.
    Paternity leave is 5 days and must be taken within 3 months of childbirth.
    Office working hours are 9 AM to 6 PM Monday to Friday.
    Work from home is allowed up to 2 days per week with manager approval.
    Annual leave entitlement is 20 days per year for all permanent employees.
    Annual leave must be applied 2 weeks in advance through the HR portal.
    """

    chunks = chunk_document(document, chunk_size=200, overlap=50)

    print(f"Total characters in document: {len(document)}")
    print(f"chunks : {chunks}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Chunk size: 200 characters, Overlap: 50 characters")
    print("\n" + "="*50 + "\n")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n {chunk}\n {'-'*50}\n")

if __name__ == "__main__":
    main()
