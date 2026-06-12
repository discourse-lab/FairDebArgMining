"""
All prompt texts are here...
"""

system_prompt = (
        "Du bist ein Deutschlehrer der 9. Klasse und analysierst argumentative Aufsätze deiner Schüler. "
        "Die Schüler schreiben den Aufsatz mit Bezug auf eine vorgegebene Streitfrage. "
        "Du überprüfst, wie der Aufsatz argumentativ aufgebaut ist. "
    )

system_prompt_extended = (
        "Das heißt, du analysierst, welche Sätze im Text argumentativ sind, "
        "und wie die argumentativen Sätze zur Streitfrage und zu einander stehen. "
)

user_prompt_task_general = (
        "Im Folgenden liegt ein Aufsatz vor. "
        "Der Aufsatz wurde in kleineren Einheiten wie Sätzen oder Phrasen unterteilt, "
        "die im folgenden zur Vereinfachung als 'Sätze' bezeichnet werden. "
        "Die Sätze sind nummeriert. Das Format dabei is 'Satznummer: Satz'.\n"
)

user_prompt_overall_constellation = (
        "Entscheide zunächst, welche argumentative Gesamtkonstellation vorliegt:\n"
        "'Entschieden' bedeutet: Der Aufsatz argumentiert für eine bestimmte Position des Autors. Dabei kann auch die "
        "Gegenposition präsentiert und begründet werden, doch der Autor entscheidet sich letztlich für eine Seite.\n"
        "'Unentschieden' bedeutet: Der Aufsatz bringt gleichberechtigt Argumente für beide Seiten der Streitfrage "
        "hervor. Der Autor entscheidet sich nicht für eine Position.\n"
)

user_prompt_zones_with_oc = (
        "Analysiere die Funktionen der einzelnen Sätze. Dabei gilt eine grundlegende Unterscheidung zwischen der "
        "Funktion der These und der des Arguments: Eine These ist eine argumentative Position, die jemand einnehmen "
        "kann. Ein Argument ist ein Grund, der eine bestimmte These stützt oder angreift.\n"
        "Jeder Satz im Aufsatz soll einer der folgenden 6 Funktionen zugeordnet werden:\n"
        "'Zentrale_These' bedeutet: "
        "Der Satz beschreibt in einem entschiedenen Aufsatz die Kernposition des Autors, oder er drückt in einem "
        "unentschiedenen Aufsatz explizit aus, dass der Autor sich nicht für eine Position entscheiden kann.\n"
        "'These_1' bedeutet: "
        "Der Satz beschreibt in einem entschiedenen Aufsatz die Position, die mit der zentralen "
        "These übereinstimmt, oder er beschreibt in einem unentschiedenen Aufsatz die im Text erstgenannte Position.\n"
        "'These_2' bedeutet: "
        "Der Satz beschreibt in einem entschiedenen Aufsatz die Position gegen die der zentralen These, "
        "oder er beschreibt in einem unentschiedenen Aufsatz die im Text später genannte Position.\n"
        "'Pro_Argument' bedeutet: "
        "Der Satz beschreibt ein Argument, das These 1, also die Position des Autors in einem entschiedenen Aufsatz, "
        "stützt und bestärkt.\n"
        "'Con_Argument' bedeutet: "
        "Der Satz beschreibt ein Argument, das These 2, also die Position gegen die des Autors in einem entschiedenen "
        "Aufsatz, stützt und bestärkt.\n"
        "'Sonstiges' bedeutet: "
        "Der Satz hat keine argumentative Funktion. Stattdessen gibt er zum Beispiel "
        "Hintergrundinformationen zum Thema, Anekdoten des Autors oder dient der Gliederung und der abschließenden"
        "Zusammenfassung des Aufsatzes.\n"
    )

user_prompt_concl_additional = (
    "Manche Aufsätze enden mit einer knappen Zusammenfassung der Position des Autors, zum Teil etwas umformuliert. "
    "Sofern die zentrale These bereits woanders markiert wurde, wird dieser Abschlusssatz als Zusammenfassung des "
    "Aufsatzes gewertet und somit als 'Sonstiges' markiert. Gibt es hingegen an keiner anderen Stelle die "
    "zentrale These, so wird dieser Abschlusssatz als 'Zentrale_These' markiert.\n"
)

user_prompt_anno_steps = (
    "Bei der Zuordnung kannst du wie folgt vorgehen:\n"
    "1. Finde die Kernaussage des Textes ('Zentrale_These'): Diese steht für sich selbst "
    "und kann prinzipiell an jeder Stelle des Textes stehen.\n"
    "2. Finde weitere Thesen ('These_1' / 'These_2'): Wenn der Text beide Positionen beleuchtet, unabhängig davon, ob "
    "der Autor sich letztendlich für eine Seite entscheidet, werden die beiden Positionen markiert.\n"
    "3. Finde Argumente ('Pro_Argument' / 'Con_Argument'): Argumente begründen die Thesen. Dabei können sie die Thesen "
    "direkt stützen oder ein anderes Argument und damit indirekt die Thesen stützen.\n"
    "4. Markiere verbleibende Sätze ('Sonstiges'): Alle Sätze, die zuvor nicht markiert wurden, haben keine "
    "argumentative Funktion.\n"
)

user_prompt_command_with_oc = (
    "Werte den Aufsatz aus und ordne jeden Satz einer der genannten 6 Funktionen zu. Das Format des Outputs "
    "soll ausschließlich wie folgt sein:\n'Gesamtkonstellation\nSatznummer: Funktion\nSatznummer: Funktion'\n, "
    "zum Beispiel\n'Entschieden\n1: Zentrale_These\n2: Sonstiges\n3: Pro_Argument\n4: These_1'.\n "
    "Im Output sollen es neben der Gesamtkonstellation genauso viele "
    "Satznummer-Funktion-Paare geben wie es Sätze im Aufsatz gibt."
    "\n\nSchüleraufsatz\n\n{}"
)

# Store in a single dict item to be imported
all_prompt_templates = {
    "system_prompt": system_prompt,
    "system_prompt_extended": system_prompt_extended,
    "user_prompt_task_general": user_prompt_task_general,
    "user_prompt_overall_constellation": user_prompt_overall_constellation,
    "user_prompt_zones_with_oc": user_prompt_zones_with_oc,
    "user_prompt_concl_additional": user_prompt_concl_additional,
    "user_prompt_anno_steps": user_prompt_anno_steps,
    # "user_prompt_zones_no_oc": user_prompt_zones_no_oc,
    "user_prompt_command_with_oc": user_prompt_command_with_oc,
    # "user_prompt_command_no_oc": user_prompt_command_no_oc
}
