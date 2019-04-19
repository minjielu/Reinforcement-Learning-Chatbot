# This is the main program for the GUI

import os
from Tkinter import *
# import PIL
import pyscreenshot as ImageGrab
from seq2seq import Evaluator


class main:

    def __init__(self, master):
        # Initialize the api
        self.roles = ['AI', 'You']
        self.curRole = 1
        # The role alternates between the AI(Artificial Intelligence) and You
        # in the chatting process.
        self.evaluator = Evaluator()
        self.chatHistory = []
        self.justStarting = True  # To specify that the chat is not started yet.
        self.master = master
        self.lastSentence = StringVar()
        self.response = StringVar()
        self.saveFile = StringVar()
                                  # The file directory to save the chat
                                  # history.
        f1 = Frame(self.master, padx=5, pady=5)
        f1.pack(side=LEFT, fill=Y)
        Label(f1, text="Chat history",
              fg="green", font=("", 12)).pack(padx = 30)
        self.list = Listbox(
            f1, bg="cyan", font=12, fg="green", width=40)
                            # The listbox to display chat history.
        self.list.pack()
        self.label1 = Label(
            f1, text="Start chatting:", fg="green", font=("", 12))
        self.label1.pack()
        Entry2 = Entry(
            f1, textvariable=self.response, width=40).pack()  # Your reply.
        Button(f1, text="Sumbit", command=self.getResult,
               fg="red", bg="blue").pack()
        self.sco = Label(
            f1, text="Your score is NULL", fg="green", font=("", 12))
        self.sco.pack()
        Label(f1, text="Recommended response:",
              fg="green", font=("", 12)).pack()
        self.rec = Label(f1, text="NULL", fg="green", font=("", 12))
        self.rec.pack()
        Button(f1, text="Save chat history to:",
               command=self.saveChat, fg="red", bg="blue").pack()
        Entry(f1, textvariable=self.saveFile).pack(pady=10)
        Button(f1, text="ReStart", command=self.reStart,
               fg="red", bg="blue").pack()

    def reStart(self):
        # Clear the chatHistory and start a new chat.
        self.chatHistory = []
        self.list.delete(0, END)
        self.justStarting = True
        self.response.set("")
        self.label1['text'] = "Start chatting:"
        self.sco['text'] = "Your score is NULL"
        self.rec['text'] = "NULL"

    def saveChat(self):
        # Save the chatHistory to file.
        with open(self.saveFile.get(), 'w') as fileName:
            for line in self.chatHistory:
                fileName.write(line + '\n')

    def getResult(self):
        # Invoke the Seq2Seq model to evaluate the users' reply and to generate
        # the next turn of the AI.
        if self.response.get() == "":
            return
        self.chatHistory.append(self.response.get())
        self.refreshChatHistory(self.chatHistory)
        score, recom, nextTurn, message = self.evaluator.evaluate(
            self.chatHistory)
        if self.justStarting:
            self.justStarting = False
            self.label1['text'] = "Your turn:"
            self.chatHistory.append(nextTurn)
        else:
            self.sco['text'] = "Your score is: " + score + message
            self.rec['text'] = recom
            self.chatHistory.append(nextTurn)
        self.refreshChatHistory(self.chatHistory)

    def refreshChatHistory(self, history):
        # Refresh the listbox which displays the chatHistory.
        self.list.insert(END, self.roles[self.curRole] + ": " + history[-1])
        self.curRole = 1 - self.curRole


if __name__ == "__main__":
    root = Tk()
    main(root)
    root.title('Chat Adviser')
    root.resizable(0, 0)
    root.mainloop()
