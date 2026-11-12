
import csv
import random
import os

random.seed(42)

os.makedirs("data", exist_ok=True)

payers = ["Medicare", "Medicaid", "Aetna", "Cigna", "UnitedHealth"]
providers = ["Hospital_A", "Hospital_B", "Clinic_A", "Clinic_B"]
cpt_codes = ["99213", "99214", "99215", "93000", "80053"]
diagnosis_codes = ["D001", "D002", "D003", "D004", "D005"]

claims = []

for i in range(1, 501):

    # Generate claim details independently
    payer = random.choice(payers)
    provider = random.choice(providers)
    cpt_code = random.choice(cpt_codes)
    diagnosis_code = random.choice(diagnosis_codes)

    billed_amount = round(random.uniform(100, 5000), 2)
    patient_age = random.randint(18, 90)

    # Create a risk score using claim characteristics
    risk = 0

    if payer in ["Medicaid", "Cigna"]:
        risk += 0.15

    if provider in ["Clinic_B"]:
        risk += 0.15

    if billed_amount > 3500:
        risk += 0.20

    if diagnosis_code in ["D004", "D005"]:
        risk += 0.15

    if patient_age > 70:
        risk += 0.05

    risk += random.uniform(0, 0.5)

    # Determine outcome with some randomness
    if risk < 0.55:
        status = "Paid"
    elif risk < 0.70:
        status = "Partially_Paid"
    else:
        status = "Denied"

    claims.append({
        "id": i,
        "payer": payer,
        "provider": provider,
        "cpt_code": cpt_code,
        "diagnosis_code": diagnosis_code,
        "billed_amount": billed_amount,
        "patient_age": patient_age,
        "status": status
    })

file_path = "data/claims.csv"

with open(file_path, "w", newline="") as file:
    fieldnames = [
        "id",
        "payer",
        "provider",
        "cpt_code",
        "diagnosis_code",
        "billed_amount",
        "patient_age",
        "status"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(claims)

print("Dataset generated successfully!")
print("Total claims:", len(claims))

for outcome in ["Paid", "Partially_Paid", "Denied"]:
    count = sum(1 for claim in claims if claim["status"] == outcome)
    print(outcome, "claims:", count)

print("Saved at:", file_path)
