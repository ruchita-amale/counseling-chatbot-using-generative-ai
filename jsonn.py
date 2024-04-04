import random

# Define question and answer templates
question_templates = [
    "How can I {action}?",
    "What should I do when I {action}?",
    "I'm struggling with {problem}. Any advice?",
    "Can you help me with {problem}?",
    "What are some tips for dealing with {problem}?",
    "How do I handle {problem}?",
    "I need advice on {problem}.",
    "What steps can I take to {action}?",
    "How can I overcome {problem}?",
    "I feel {emotion} when I {action}. What should I do?"
]

answer_templates = [
    "You can {solution}.",
    "Try {solution}.",
    "Consider {solution} as a solution.",
    "Here's a suggestion: {solution}.",
    "{solution} might help.",
    "One way to address this is by {solution}.",
    "Have you tried {solution}?",
    "I recommend {solution}.",
    "{solution} could be beneficial.",
    "It might be helpful to {solution}."
]

# Define lists of actions, problems, emotions, and solutions
actions = ["manage stress", "improve self-esteem", "stop overthinking", "communicate with my partner", "connect with others",
           "cope with loss", "stay motivated", "calm my nerves", "create work-life balance", "find happiness"]
problems = ["anxiety", "depression", "loneliness", "relationship issues", "work-related stress", "financial difficulties",
            "self-doubt", "procrastination", "anger management", "family conflicts"]
emotions = ["anxious", "sad", "frustrated", "stressed", "overwhelmed", "confused", "angry", "disappointed", "lonely", "hopeless"]
solutions = ["practice mindfulness", "seek professional help", "engage in self-care activities", "set boundaries",
             "talk to a trusted friend or family member", "seek therapy", "try relaxation techniques", "establish a routine",
             "prioritize tasks", "take breaks regularly"]

# Generate question-answer pairs
question_answer_pairs = []
for _ in range(1000):
    question_template = random.choice(question_templates)
    answer_template = random.choice(answer_templates)
    action = random.choice(actions)
    problem = random.choice(problems)
    emotion = random.choice(emotions)
    solution = random.choice(solutions)

    question = question_template.format(action=action, problem=problem, emotion=emotion)
    answer = answer_template.format(solution=solution)

    question_answer_pairs.append({"question": question, "answer": answer})

# Write to JSON file
import json

with open("question_answer_pairs.json", "w") as json_file:
    json.dump(question_answer_pairs, json_file, indent=4)
