class BugTracker:
    def __init__(self):
        self.bugs = []

    def add_bug(self,bug_id, description, severity):
        bug = {
            "id": bug_id,
            "description": description,
            "severity": severity,
            "status": "Open"
        }
        self.bugs.append(bug)
        print(f"Bug {bug_id} added successfully.")

    def update_bug(self, bug_id, new_status):
        if bug_id in self.bugs:
            self.bugs[bug_id]["status"] = new_status
            print(f"Bug {bug_id} status updated to {new_status}.")
        else:
            print(f"Bug {bug_id} not found.")
    def list_bugs(self):
        if not self.bugs:
            print("No bugs reported.")
        else:
            for bug in self.bugs:
             print(f"Bug ID: {bug['id']}")
             print(f"  Description: {bug['description']}")
             print(f"  Severity: {bug['severity']}")
             print(f"  Status: {bug['status']}")
             print("--------------------------")

if __name__ == "__main__":
    tracker = BugTracker()
    tracker.add_bug(1, "Login button not working", "High")
    tracker.add_bug(2, "Typo in homepage", "Low")
    tracker.list_bugs()
    tracker.update_bug(1, "In Progress")
    tracker.list_bugs()