import pandas as pd
import streamlit as st
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"],
    "Department": ["HR", "IT", "Finance", "IT", "Finance", "HR", "IT", "HR", "Finance", "IT"],
    "Salary": [48000, 60000, 52000, 70000, 49000, 55000, 51000, 47000, 65000, 80000],
    "Experience": [2, 3, 6, 4, 1, 7, 2, 4, 5, 6]
}
df = pd.DataFrame(data)
print("First 5 rows: ")
print(df.head())
print("\nLast 5 rows: ")
print(df.tail())
print("\nColumn names: ")
print(df.columns)
print("\nData types: ")
print(df.dtypes)
grouped = df.groupby("Department")[["Salary", "Experience"]].mean()
print("\nAverage Salary and Experience by Department:")
print(grouped)
filtered=df[(df["Salary"]>50000)&(df["Experience"]<5)]
print("\n Employees with salary>50000 and Experience<5 years: ")
print(filtered)
print(df.columns)
st.text_input("Title and Desription")
st.subheader("Filtered data (Salary > 50000 and Experience < 5 years)")
st.write(filtered)
st.subheader("Average salary by department")
st.bar_chart(grouped["Salary"])
csv = filtered.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name='filtered_employees.csv',
    mime='text/csv'
)