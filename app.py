import numpy as np
import pandas as pd
import random
import os

# Ensure scikit-learn is installed
try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    print("Error: scikit-learn is not installed. Run 'pip install scikit-learn' and try again.")
    exit()

# 🎓 Student Class
class Student:
    def __init__(self, name):
        self.name = name
        self.level = 1
        self.xp = 0
        self.next_level_xp = 100

    def gain_xp(self, points):
        self.xp += points
        print(f"🎉 {self.name} gained {points} XP!")
        self.check_level_up()

    def check_level_up(self):
        while self.xp >= self.next_level_xp:
            self.level_up()

    def level_up(self):
        self.level += 1
        self.xp -= self.next_level_xp
        self.next_level_xp = int(self.next_level_xp * 1.5)
        print(f"🚀 {self.name} leveled up! Now at level {self.level}.")

# 📊 Create simulated student data
num_samples = 10
data = {
    "visual_spatial": [random.randint(1, 5) for _ in range(num_samples)],
    "linguistic_verbal": [random.randint(1, 5) for _ in range(num_samples)],
    "logical_mathematical": [random.randint(1, 5) for _ in range(num_samples)],
    "bodily_kinesthetic": [random.randint(1, 5) for _ in range(num_samples)],
    "musical": [random.randint(1, 5) for _ in range(num_samples)],
    "interpersonal": [random.randint(1, 5) for _ in range(num_samples)],
    "intrapersonal": [random.randint(1, 5) for _ in range(num_samples)],
    "naturalistic": [random.randint(1, 5) for _ in range(num_samples)],
    "intelligence_type": [
        "Visual-Spatial", "Linguistic-Verbal", "Logical-Mathematical",
        "Bodily-Kinesthetic", "Musical", "Interpersonal",
        "Intrapersonal", "Naturalistic", "Logical-Mathematical", "Musical"
    ]
}
df = pd.DataFrame(data)

# 📊 Prepare training data
X = df.drop(columns=["intelligence_type"])
y = df["intelligence_type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌱 Train the Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 📊 Evaluate model accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# 📝 Function to get student input from the user
def get_student_input():
    print("📝 Answer each question from 1 (low) to 5 (high).")
    return {
        "visual_spatial": int(input("How well do you learn through images and diagrams? (1-5): ")),
        "linguistic_verbal": int(input("Do you prefer reading and writing? (1-5): ")),
        "logical_mathematical": int(input("Are you good at problem-solving and puzzles? (1-5): ")),
        "bodily_kinesthetic": int(input("Do you learn best by moving and doing? (1-5): ")),
        "musical": int(input("Does music help you learn? (1-5): ")),
        "interpersonal": int(input("Do you enjoy working with others? (1-5): ")),
        "intrapersonal": int(input("Do you prefer self-reflection and independent learning? (1-5): ")),
        "naturalistic": int(input("Are you drawn to nature and the environment? (1-5): "))
    }

student_data = get_student_input()

# 🔍 Predict the student's intelligence type based on input
def predict_intelligence(student_input):
    input_data = np.array([list(student_input.values())])
    return model.predict(input_data)[0]

intelligence_type = predict_intelligence(student_data)
print(f"🔍 Predicted Intelligence Type: {intelligence_type}")

# 📖 Provide a personalized learning recommendation
def learning_recommendation(intelligence_type):
    recommendations = {
        "Visual-Spatial": "📚 Use diagrams, charts, and visual aids.",
        "Linguistic-Verbal": "📚 Engage in reading, writing, and storytelling.",
        "Logical-Mathematical": "📚 Solve puzzles, play strategy games, and study logic.",
        "Bodily-Kinesthetic": "📚 Learn through hands-on activities and physical movement.",
        "Musical": "📚 Use music, rhythm, and sound in your learning.",
        "Interpersonal": "📚 Work in groups, collaborate, and discuss with others.",
        "Intrapersonal": "📚 Reflect, set personal goals, and study independently.",
        "Naturalistic": "📚 Explore nature, conduct experiments, and study biological sciences."
    }
    return recommendations.get(intelligence_type, "🔍 Try exploring multiple learning styles!")

lesson_plan = learning_recommendation(intelligence_type)
print(f"📚 Personalized Learning Path: {lesson_plan}")

# 🏆 Gamification: Simulate a student gaining XP
if __name__ == "__main__":
    student = Student("Alex")
    student.gain_xp(random.randint(30, 70))
    student.gain_xp(random.randint(40, 80))
    
    # Save the student's results to a CSV file (append mode)
    df_results = pd.DataFrame([student_data])
    df_results["intelligence_type"] = intelligence_type
    df_results.to_csv(
        "student_results.csv",
        mode="a",
        index=False,
        header=not os.path.exists("student_results.csv")
    )
