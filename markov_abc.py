import numpy as np
from fractions import Fraction

ABC_TEXT = r"""
X: 1
T: 1. Classical Minuet
M: 3/4
L: 1/8
K: G
|: "G" G2 B2 d2 | "D7" c2 A2 F2 | "G" G2 B2 d2 | "C" e4 c2 |
|  "G" d2 B2 G2 | "D7" A2 F2 D2 | "G" B,2 D2 G2 | "G" G6 :|

X: 2
T: 2. Lively Jig
M: 6/8
L: 1/8
K: D
|: "D" DFA AFA | "G" GAG GAG | "D" DFA AFA | "A7" EGE EGE |
|  "D" DFA AFA | "G" GAG GAG | "A7" Ace gfe | "D" d3 d3 :|

X: 3
T: 3. Slow Blues
M: 4/4
L: 1/8
K: A
|: "A7" C2 E2 G2 A2 | c4 B2 A2 | "D7" F2 A2 d2 c2 | B4 A2 G2 |
|  "A7" C2 E2 G2 A2 | "E7" E2 B,2 ^G,2 B,2 | "A7" A,8- | A,8 :|

X: 4
T: 4. Pop Chorus
M: 4/4
L: 1/8
K: C
|: "C" c2 c2 e2 g2 | "G" d2 d2 B2 d2 | "Am" c2 c2 e2 a2 | "F" g2 f2 e2 d2 |
|  "C" c2 c2 e2 g2 | "G" d2 B2 G2 B2 | "F" A2 F2 C2 F2 | "C" G8 :|

X: 5
T: 5. Viennese Waltz
M: 3/4
L: 1/8
K: F
|: "F" c'2 af c2 | "C7" b2 ge B2 | "F" a2 fc A2 | "Bb" g4 f2 |
|  "F" a2 fc A2 | "C7" ge B2 G2 | "F" F2 A2 c2 | "F" f6 :|

X: 6
T: 6. Swing Jazz Lick
M: 4/4
L: 1/8
K: Bb
|: "Bbmaj7" d2 f2 a2 g2 | "Gm7" f2 d2 B2 d2 | "Cm7" e2 g2 b2 a2 | "F7" g2 e2 c2 A2 |
|  "Bbmaj7" d4- d2 B2 | "F7" A,2 C2 E2 F2 | "Bb6" G8- | G8 :|

X: 7
T: 7. Sea Shanty
M: 2/4
L: 1/8
K: Dm
|: "Dm" A>G FD | "C" E2 G2 | "Dm" A>G FD | "A7" E4 |
|  "Dm" d>c AF | "Gm" G2 B2 | "A7" cB AG | "Dm" D4 :|

X: 8
T: 8. Bossa Nova Groove
M: 4/4
L: 1/8
K: Am
|: "Am7" c2 B2 A2 G2 | E4 z4 | "Dm7" f2 e2 d2 c2 | B4 z4 |
| "G7" B2 d2 g2 f2 | e4 z4 | "Cmaj7" c2 e2 g2 a2 | g8 :|

X: 9
T: 9. Irish Reel
M: 4/4
L: 1/8
K: Em
|: "Em" e2 BE GBEB | "D" d2 AD FADF | "Em" e2 BE GBEG | "Bm" BAFB dAFB |
|  "Em" e2 BE GBEB | "D" d2 AD FADF | "G" GABG "A" ABcA | "B7" BGEG FDEF :|

X: 10
T: 10. Gentle Lullaby
M: 3/4
L: 1/8
K: F
|: "F" F2 A2 c2 | "Bb" d4 B2 | "F" c2 A2 F2 | "C7" G4 G2 |
|  "F" F2 A2 c2 | "Bb" d4 B2 | "C7" c2 B2 G2 | "F" F6 :|

X: 11
T: 11. Toy Soldier March
M: 2/4
L: 1/8
K: C
|: "C" C E G E | "F" F A c A | "C" G E G E | "G7" D F B, D |
|  "C" C E G c | "F" F A c f | "G7" g f e d | "C" c4 :|

X: 12
T: 12. Funk Riff
M: 4/4
L: 1/8
K: Em
|: "Em7" E,2 ^G, E, B,, E,2 G, B, | E,2 ^G, E, B,, ^G,4 | "A7" A,2 ^C A, E, A,2 C E | A,2 ^C A, E, ^C4 :|

X: 13
T: 13. Power Ballad Intro
M: 4/4
L: 1/8
K: G
|: "G" g4 f2 e2 | "D/F#" d6 B2 | "Em" e4 d2 c2 | "C" c6 G2 |
|  "G" B4 A2 G2 | "D" A6 F2 | "C" G4- G4 | G8 :|

X: 14
T: 14. Country Twang
M: 4/4
L: 1/8
K: A
|: "A" A,2 C2 E2 A2 | c4 e4 | "D" d2 f2 a2 f2 | "A" e4 c4 |
|  "A" A,2 C2 E2 A2 | "E7" B,2 D2 ^G2 B2 | "A" A8- | A8 :|

X: 15
T: 15. Spooky Theme
M: 4/4
L: 1/8
K: Cm
|: "Cm" c2 d2 _e2 f2 | "G" g4- g2 f2 | "Fm" _a2 g2 f2 _e2 | "Cm" d4- d2 c2 |
|  "Ab" _e2 f2 g2 _a2 | "G7" b4 g4 | "Cm" c8- | c8 :|

X: 16
T: 16. Baroque Fanfare
M: 4/4
L: 1/8
K: D
|: "D" d2 A2 F2 A2 | "G" B2 G2 E2 G2 | "D" A2 F2 D2 F2 | "A7" E2 C2 A,2 C2 |
|  "D" d2 A2 F2 d2 | "G" g2 B2 G2 e2 | "A7" a2 c2 A2 f2 | "D" d8 :|

X: 17
T: 17. Tango Rhythm
M: 4/4
L: 1/8
K: Am
|: "Am" A2 A,2 C2 E2 | "E7" ^G4 F2 E2 | "Dm" D2 D,2 F2 A2 | "Am" E4 C4 |
|  "Am" c2 B2 A2 G2 | "E7" ^G2 A2 B2 ^c2 | "Am" A8- | A8 :|

X: 18
T: 18. Reggae Skank
M: 4/4
L: 1/8
K: G
|: z2 "G" G2 z2 "G" G2 | z2 "C" E2 z2 "C" E2 | z2 "G" D2 z2 "G" D2 | z2 "D" D2 z2 "D" F2 |
|  z2 "G" G2 z2 "G" B2 | z2 "C" c2 z2 "C" E2 | z2 "G" D2 "D7" C2 | "G" B,4 z4 :|

X: 19
T: 19. Simple Hymn
M: 4/4
L: 1/8
K: C
|: "C" C2 E2 G2 c2 | "F" A4 "C" G4 | "G7" G2 F2 E2 D2 | "C" C4 z4 :|
|: "Am" A,2 C2 E2 A2 | "E7" B4 ^G4 | "F" A2 G2 F2 E2 | "C" G4 "G7" F4 | "C" E8- | E8 :|

X: 20
T: 20. Classic Rock Riff
M: 4/4
L: 1/8
K: E
|: "E5" E,2 E,2 B,, E, B,, E, | "A5" A,,2 A,,2 E, A,, E, A,, | "E5" E,2 E,2 B,, E, B,, E, | "B5" B,,2 B,,2 ^F, B,, ^F, B,, :|

X: 21
T: 21. Romantic Nocturne
M: 4/4
L: 1/8
K: Eb
|: "Eb" G3 A B2 c2 | "Ab" _d4 c4 | "Fm" c3 B _A2 G2 | "Bb7" F8 |
|  "Eb" g3 f e2 d2 | "Cm" c6 B2 | "Ab" _A2 c2 "Bb7" d2 f2 | "Eb" e8 :|

X: 22
T: 22. Polka Dance
M: 2/4
L: 1/8
K: G
|: "G" G>A Bd | "C" c2 e2 | "G" d>B GD | "D7" c2 A2 |
|  "G" G>A Bd | "C" c2 e2 | "D7" d>c AF | "G" G4 :|

X: 23
T: 23. Medieval Chant
M: 4/4
L: 1/8
K: Ddor
|: "Dm" D2 F2 G2 A2 | "C" G4 F4 | "Dm" D2 E2 F2 D2 | "C" E4 "G" D4 |
|  "F" F2 G2 A2 c2 | "C" G4 F4 | "Dm" A2 G2 F2 E2 | "Dm" D8 :|

X: 24
T: 24. Salsa Montuno
M: 4/4
L: 1/8
K: Gm
|: "Gm" G2 B2 d2 B2 | "Cm" c2 _e2 g2 _e2 | "D7" d2 ^f2 a2 ^f2 | "Gm" g2 d2 B2 G2 :|
|: "Cm" c_e gc _e g c_e | "Gm" Bd gd B d gB | "D7" d^fa^f d^f a^f | "Gm" gbag fd_ec :|

X: 25
T: 25. Bluegrass Breakdown
M: 2/2
L: 1/8
K: A
|: "A" cBAc EAGB | "D" Addc defg | "A" aecA EAcE | "E7" BEGB EBGE |
|  "A" cBAc EAGB | "D" Addc defa | "A" gecA "E7" GABG | "A" A2 A4 :|

X: 26
T: 26. Gospel Praise
M: 4/4
L: 1/8
K: F
|: "F" F2 A2 c3 d | "Bb" d6 B2 | "F" c2 A2 F2 A2 | "C7" G4 z4 |
|  "F" A3 G F2 A2 | "Bb" d4 c2 B2 | "F/C" c2 "C7" B2 G2 | "F" F4 z4 :|

X: 27
T: 27. Cinematic Theme
M: 4/4
L: 1/8
K: Dm
|: "Dm" d4 A4 | "Bb" f4 d4 | "Gm" g4 e4 | "A7" ^c8 |
|  "Dm" d2 f2 a2 g2 | "Bb" f2 d2 B2 G2 | "C" E2 G2 c2 B2 | "A7" A8 :|

X: 28
T: 28. Calypso Melody
M: 4/4
L: 1/8
K: C
|: "C" c2 G2 c2 G2 | e2 d2 c4 | "G7" d2 G2 d2 G2 | f2 e2 d4 |
|  "C" c2 G2 e2 g2 | "F" f2 e2 d2 c2 | "G7" B2 d2 G2 F2 | "C" C8 :|

X: 29
T: 29. Folk Dance (Hora)
M: 3/8
L: 1/16
K: E phrygian
|: "Am" E2 ^G2 B2 | "G" G2 F2 E2 | "Dm" D2 E2 F2 | "E" E4 z2 :|
|: "Am" c2 B2 A2 | "G" B2 A2 G2 | "F" A2 G2 F2 | "E" E4 z2 :|

X: 30
T: 30. Soul Groove
M: 4/4
L: 1/8
K: Am
|: "Am7" A,2 C2 E2 A2 | c4 B2 A2 | "G7" G,2 B,2 D2 G2 | B4 A2 G2 :|
|: "Fmaj7" F,2 A,2 C2 F2 | A4 G2 F2 | "E7" E,2 ^G,2 B,2 D2 | E8 :|

X: 31
T: 31. Mazurka
M: 3/4
L: 1/8
K: Bb
|: "Bb" B>c d2 B2 | "F7" A>B c2 A2 | "Bb" d2 B2 F2 | "Eb" G4 G2 |
|  "Bb" B>c d2 B2 | "F7" A>B c2 A2 | "F7" c2 A2 F2 | "Bb" B6 :|

X: 32
T: 32. Bossa Nova Melody
M: 4/4
L: 1/8
K: F
|: "Fmaj7" f3 g a2 g2 | "Gm7" g4 z4 | "C7" c'3 b a2 g2 | "Fmaj7" a4 z4 :|

X: 33
T: 33. Sad Folk Song
M: 3/4
L: 1/8
K: Gm
|: "Gm" G2 A2 B2 | "D7" d4 c2 | "Gm" B2 A2 G2 | "Cm" c4 B2 |
|  "Gm" G2 d2 B2 | "D7" A4 G2 | "Gm" G6- | G6 :|

X: 34
T: 34. Cheerful Folk Tune
M: 2/4
L: 1/8
K: D
|: "D" FA df | "G" g2 B2 | "D" af ed | "A7" c4 |
|  "D" FA df | "G" g2 B2 | "A7" ce ge | "D" d4 :|

X: 35
T: 35. Smooth Jazz Lick
M: 4/4
L: 1/8
K: Cm
|: "Cm7" c2 _e2 g2 f2 | "Fm7" _a2 g2 f2 _e2 | "Bb7" d2 f2 _a2 g2 | "Ebmaj7" b2 g2 _e2 d2 :|

X: 36
T: 36. Strathspey
M: 4/4
L: 1/8
K: Amix
|: "A" A>B c<e | "G" d<G B>G | "A" A>B c<e | "E" e>d c<B |
|  "A" A>B c<e | "G" d<G B>G | "A" c>A "G" B>G | "A" A4 A2 :|

X: 37
T: 37. Courtly Dance (Pavane)
M: 4/4
L: 1/8
K: Dm
|: "Dm" D3 E F2 G2 | "C" E4 D4 | "Gm" G3 A B2 c2 | "A7" A4 G4 |
|  "Dm" F2 E2 D2 C2 | "Bb" D4 C4 | "A7" E2 ^C2 A,2 C2 | "Dm" D8 :|

X: 38
T: 38. Fast Bebop Line
M: 4/4
L: 1/8
K: F
|: "F" FAcf agfa | "Gm7" gfed "C7" cBAG | "Am7" Acea "D7" bagf | "Gm7" edcB "C7" AGFE :|

X: 39
T: 39. Film Noir Theme
M: 4/4
L: 1/8
K: Cm
|: "Cm" c2 G,2 C2 G,2 | _E2 D2 C4 | "G7" B,2 ^F,2 B,2 ^F,2 | G,2 F,2 E,4 :|

X: 40
T: 40. Island Steel Pan Tune
M: 4/4
L: 1/8
K: G
|: "G" g2 d2 B2 d2 | "C" c'2 g2 e2 g2 | "D7" a2 f2 d2 f2 | "G" g4 z4 :|

X: 41
T: 41. Ambient Melody
M: 4/4
L: 1/8
K: Am
|: "Am" A,8 | "G" G,,8 | "Fmaj7" F,,8 | "E" E,,8 :|

X: 42
T: 42. Upbeat Ska Riff
M: 4/4
L: 1/8
K: C
|: z C E G z C E G | z F A c z F A c | z C E G z C E G | z G, B, D z G, B, D :|

X: 43
T: 43. French Musette Waltz
M: 3/4
L: 1/8
K: C
|: "C" e2 d2 c2 | "G7" B2 A2 G2 | "C" c2 B2 A2 | G4 E2 |
| "F" f2 e2 d2 | "C" e2 d2 c2 | "G7" B2 d2 G2 | "C" c6 :|

X: 44
T: 44. Klezmer Tune
M: 2/4
L: 1/8
K: Dm
|: "Dm" D>E FG | "A7" A_B =c_B | "Gm" B>c d_e | "A7" =c4 |
|  "Dm" f>e d^c | "Gm" d2 cB | "A7" AG FE | "Dm" D4 :|

X: 45
T: 45. Hard Rock Power Chords
M: 4/4
L: 1/8
K: Dm
|: "D5" D,2 D, A,, D, A,, D, A,, | "C5" C,2 C, G,, C, G,, C, G,, | "G5" G,,2 G,, D, G,, D, G,, D, | "D5" D,2 D, A,, D, A,, D, A,, :|

X: 46
T: 46. Simple Nursery Rhyme
M: 2/4
L: 1/8
K: F
|: "F" C C F F | A A G2 | "Bb" F F E E | "C7" D D C2 :|

X: 47
T: 47. Sci-Fi Theme
M: 4/4
L: 1/8
K: Cm
|: "Cm" c_e g_b | "Ab" c'_a f_e | "G" d=b g=f | "Cm" ec Gc :|

X: 48
T: 48. Disco Bassline
M: 4/4
L: 1/8
K: Am
|: "Am" A,A, E,A, A,A, E,A, | "G" G,,G,, D,G,, G,,G,, D,G,, | "F" F,,F,, C,F,, F,,F,, C,F,, | "E" E,,E,, B,,E,, E,,E,, B,,E,, :|

X: 49
T: 49. Celtic Air
M: 3/4
L: 1/8
K: G
|: "G" d2 B2 G2 | "C" e3 d c2 | "G" B2 G2 E2 | "D" A6 |
|  "Em" G2 B2 e2 | "C" d3 c B2 | "G" G2 E2 "D7" D2 | "G" G6 :|

X: 50
T: 50. Grand Finale Fanfare
M: 4/4
L: 1/8
K: C
|: "C" C2G2 c4 | "G" G2d2 g4 | "C" c2e2 g2e2 | "F" a4 "G7" g4 |
|  "C" c8- | c4 "G7" d4 | "C" e4 f4 | "G7" g4 a4 | "C" c'8- | c'8 :|
""".strip("\n")


def split_tunes(abc_text):
    blocks, curr = [], []
    for line in abc_text.splitlines():
        if line.startswith("X:"):
            if curr:
                blocks.append("\n".join(curr))
                curr = []
        curr.append(line)
    if curr:
        blocks.append("\n".join(curr))
    return blocks


def extract_field(block, tag):
    for line in block.splitlines():
        if line.startswith(tag + ":"):
            return line[len(tag)+1:].strip()
    return ""


def extract_music_body(block):
    lines = block.splitlines()
    music_lines, k_seen = [], False
    for line in lines:
        if k_seen:
            music_lines.append(line)
        elif line.startswith("K:"):
            k_seen = True
    return " ".join(music_lines).strip()


# chars
BAR_CHARS = set("|:[]")
IGNORED_CHARS = set("()")
ACCIDENTALS = set("^_=")
OCTAVE_CHARS = set("',")
NOTE_LETTERS = set("ABCDEFGabcdefg")
REST_LETTERS = set("zZ")
TIE_CHAR = "-"
DIGITS = set("0123456789")

def parse_music_stream(music_str, base_len_str):

    i = 0
    n = len(music_str)

    curr_chord = None
    pitches, rhythms, chords = [], [], []

    # To support broken rhythm (> or <), we allow pending adjustments:
    pending_brk = None  # (op, count), op in {">","<"}

    def _apply_broken(prev_len_str, next_len_str, op, count):
        # Only adjust if both were default "1". Otherwise leave them.
        if prev_len_str != "1" or next_len_str != "1":
            return prev_len_str, next_len_str

        # '>' makes long-short, '<' makes short-long.
        # For n times '>' we use:
        # left = 2 - 1/2^n ; right = 1/2^n
        nrep = count
        half_pow = Fraction(1, 2**nrep)
        if op == ">":
            L = Fraction(2,1) - half_pow
            R = half_pow
        else:  # "<"
            L = half_pow
            R = Fraction(2,1) - half_pow

        def f2s(fr):
            return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"

        return f2s(L), f2s(R)

    def append_event(pitch_tok, length_tok, chord_tok, had_digits):
        nonlocal pending_brk
        # If we had a pending '>' or '<', adjust prev+this durations
        if pending_brk and len(rhythms) >= 1:
            op, count = pending_brk
            prev_len = rhythms[-1]
            prev_had_digits = getattr(append_event, "_prev_had_digits", False)

            if (not prev_had_digits) and (not had_digits):
                adj_prev, adj_curr = _apply_broken(prev_len, length_tok, op, count)
                rhythms[-1] = adj_prev
                length_tok = adj_curr
            pending_brk = None

        pitches.append(pitch_tok)
        rhythms.append(length_tok)
        chords.append(chord_tok if chord_tok is not None else "_NOC_")
        append_event._prev_had_digits = had_digits

    while i < n:
        ch = music_str[i]

        # skip whitespace / barlines / parentheses
        if ch.isspace() or ch in BAR_CHARS or ch in IGNORED_CHARS:
            i += 1
            continue

        # chord labels in quotes
        if ch == '"':
            j = i + 1
            while j < n and music_str[j] != '"':
                j += 1
            if j < n:
                curr_chord = music_str[i+1:j].strip()
                i = j + 1
            else:
                i += 1
            continue

        if ch in (">", "<"):
            op = ch
            cnt = 0
            while i < n and music_str[i] == op:
                cnt += 1
                i += 1
            pending_brk = (op, cnt)
            continue

        acc = ""
        while i < n and music_str[i] in ACCIDENTALS:
            acc += music_str[i]
            i += 1
        if i >= n:
            break

        ch = music_str[i]

        if ch in NOTE_LETTERS or ch in REST_LETTERS:
            note_char = music_str[i]
            i += 1

            octv = ""
            while i < n and music_str[i] in OCTAVE_CHARS:
                octv += music_str[i]
                i += 1

            dur_str = ""
            while i < n and music_str[i] in DIGITS:
                dur_str += music_str[i]
                i += 1
            had_digits = (dur_str != "")

            while i < n and music_str[i] == TIE_CHAR:
                i += 1

            pitch_tok = "z" if note_char in REST_LETTERS else f"{acc}{note_char}{octv}"
            length_tok = dur_str if dur_str else "1"

            append_event(pitch_tok, length_tok, curr_chord, had_digits)

        else:
            i += 1

    return pitches, rhythms, chords


def build_markov(seqs,
                 alpha=0.05,
                 add_start_end=True,
                 min_unigram_for_prior=3,
                 exclude_pred=None):

    START = "<START>"
    END = "<END>"

    # vocab
    vocab = set()
    for seq in seqs:
        for tok in seq:
            vocab.add(tok)
    if add_start_end:
        vocab.update([START, END])

    idx2tok = sorted(vocab)
    tok2idx = {t: i for i, t in enumerate(idx2tok)}
    N = len(idx2tok)

    C = np.zeros((N, N), dtype=float)
    unig = np.zeros(N, dtype=float)

    for seq in seqs:
        chain = [START] + seq + [END] if add_start_end else list(seq)
        for a, b in zip(chain[:-1], chain[1:]):
            ia = tok2idx[a]
            ib = tok2idx[b]
            C[ia, ib] += 1.0
            unig[ib] += 1.0

    is_meta = np.array([t in (START, END) for t in idx2tok], dtype=bool)

    if exclude_pred is None:
        exclude_mask = np.zeros(N, dtype=bool)
    else:
        exclude_mask = np.array([exclude_pred(t) for t in idx2tok], dtype=bool)

    prior_mask = (~is_meta) & (~exclude_mask) & (unig >= min_unigram_for_prior)
    if prior_mask.any():
        q = np.zeros(N, dtype=float)
        q_vals = unig[prior_mask]
        q[prior_mask] = q_vals / q_vals.sum()
    else:
        alt_mask = (~is_meta) & (~exclude_mask)
        q = np.zeros(N, dtype=float)
        if alt_mask.any():
            q[alt_mask] = 1.0 / alt_mask.sum()

    for i in range(N):
        if is_meta[i]:
            continue
        C[i, :] += alpha * q

    row_sums = C.sum(axis=1, keepdims=True)
    P = C / np.where(row_sums == 0, 1.0, row_sums)

    return P, idx2tok, tok2idx


def extract_chord_sequence_for_tune(chord_events):
    seq = []
    prev = None
    for ch in chord_events:
        if ch == "_NOC_":
            continue
        if ch != prev:
            seq.append(ch)
            prev = ch
    return seq


def train_models(abc_text):
    tunes = split_tunes(abc_text)

    pitch_seqs = []
    rhythm_seqs = []
    chord_event_seqs = []
    chord_bar_seqs = []

    for block in tunes:
        L = extract_field(block, "L")
        music = extract_music_body(block)
        if not music:
            continue

        p, r, c = parse_music_stream(music, L)

        if p:
            pitch_seqs.append(p)
            rhythm_seqs.append(r)
        if c:
            chord_event_seqs.append(c)
            chord_bar_seqs.append(extract_chord_sequence_for_tune(c))


    P_pitch, idx2pitch, pitch2idx = build_markov(
        pitch_seqs,
        alpha=0.05,
        add_start_end=True,
        min_unigram_for_prior=3,
        exclude_pred=lambda t: t == "z"
    )


    P_rhy, idx2rhy, rhy2idx = build_markov(
        rhythm_seqs,
        alpha=0.05,
        add_start_end=True,
        min_unigram_for_prior=3
    )


    P_chord, idx2chord, chord2idx = build_markov(
        chord_bar_seqs,
        alpha=0.05,
        add_start_end=True,
        min_unigram_for_prior=2
    )

    return {
        "pitch":  {"P": P_pitch,  "idx2tok": idx2pitch, "tok2idx": pitch2idx},
        "rhythm": {"P": P_rhy,    "idx2tok": idx2rhy,   "tok2idx": rhy2idx},
        "chord":  {"P": P_chord,  "idx2tok": idx2chord, "tok2idx": chord2idx},
    }


def _real_indices(idx2tok):
    return [i for i,t in enumerate(idx2tok) if t not in ("<START>","<END>")]

def _mask_and_sample(row, allowed_idx):
    probs = np.zeros_like(row)
    probs[allowed_idx] = row[allowed_idx]
    s = probs.sum()
    if s <= 0:
        probs[allowed_idx] = 1.0 / max(1,len(allowed_idx))
    else:
        probs /= s
    return np.random.choice(np.arange(len(probs)), p=probs)

def generate_rhythm_bar(models_rhythm, target_eighths, max_attempts=200):
    P = models_rhythm["P"]
    idx2tok = models_rhythm["idx2tok"]
    tok2idx = models_rhythm["tok2idx"]

    start_idx = tok2idx["<START>"]
    real_idx = _real_indices(idx2tok)

    int_idx = [
        i for i,tok in enumerate(idx2tok)
        if tok not in ("<START>","<END>") and tok.isdigit()
    ]

    if not int_idx:
        raise RuntimeError("No integer rhythm tokens found.")

    for _ in range(max_attempts):
        remain = target_eighths
        curr = start_idx
        seq = []
        ok = True
        while remain > 0:
            allowed_now = []
            for j in int_idx:
                val = int(idx2tok[j])
                if 1 <= val <= remain:
                    allowed_now.append(j)

            if not allowed_now:
                ok = False
                break

            j = _mask_and_sample(P[curr], allowed_now)
            seq.append(idx2tok[j])
            remain -= int(idx2tok[j])
            curr = j

        if ok and remain == 0:
            return seq

    raise RuntimeError("Failed to fill bar with integer rhythms.")

def generate_rhythms(models_rhythm, N_bars, meter):
    if meter == "3/4":
        per_bar = 6
    elif meter == "4/4":
        per_bar = 8
    else:
        raise ValueError("Meter must be '3/4' or '4/4'.")

    bars = []
    for _ in range(N_bars):
        bar_seq = generate_rhythm_bar(models_rhythm, per_bar)
        bars.append(bar_seq)

    flat = [x for bar in bars for x in bar]
    return bars, flat
def generate_chord_sequence(models_chord, num_bars):
    P = models_chord["P"]
    idx2tok = models_chord["idx2tok"]
    tok2idx = models_chord["tok2idx"]

    start_idx = tok2idx["<START>"]
    real_idx = [i for i, t in enumerate(idx2tok) if t not in ("<START>", "<END>")]

    seq = []
    if "C" in tok2idx:
        first_chord = "C"
        curr = tok2idx["C"]
    else:
        c_like = [t for t in idx2tok if t not in ("<START>", "<END>") and t.startswith("C")]
        if c_like:
            first_chord = c_like[0]
            curr = tok2idx[first_chord]
        else:
            j = _mask_and_sample(P[start_idx], real_idx)
            first_chord = idx2tok[j]
            curr = j

    seq.append(first_chord)


    for _ in range(num_bars - 1):
        j = _mask_and_sample(P[curr], real_idx)
        chord_tok = idx2tok[j]
        seq.append(chord_tok)
        curr = j

    return seq


def chord_root_and_quality(chord_sym):
    if not chord_sym:
        return None, None
    root = chord_sym[0]
    i = 1
    if i < len(chord_sym) and chord_sym[i] in ('b', '#'):
        root += chord_sym[i]
        i += 1
    quality = chord_sym[i:]
    return root, quality

def semitone_steps_for_quality(quality):
    q = quality.lower()

    if "maj7" in q:
        return [0,4,7,11]     # 1 3 5 7
    if q.startswith("m7"):
        return [0,3,7,10]     # 1 b3 5 b7
    if q.startswith("m"):
        return [0,3,7]        # 1 b3 5
    if q == "7":
        return [0,4,7,10]     # 1 3 5 b7
    if q.startswith("maj"):
        return [0,4,7]        # 1 3 5
    return [0,4,7]            # default major triad

sharp_ladder = ["C","^C","D","^D","E","F","^F","G","^G","A","^A","B"]
flat_ladder  = ["C","_D","D","_E","E","F","_G","G","_A","A","_B","B"]

def ladder_index(ladder, token):
    for k,v in enumerate(ladder):
        if v == token:
            return k
    return None

def abc_pitchclass_for_root(root):
    letter = root[0].upper()
    accidental = root[1:] if len(root) > 1 else ""
    if accidental == "b":
        return "_" + letter
    elif accidental == "#":
        return "^" + letter
    else:
        return letter

def transpose_pitchclass(pc, semitones):
    use_flat = "_" in pc
    ladder = flat_ladder if use_flat else sharp_ladder
    idx = ladder_index(ladder, pc)
    if idx is None:
        # try other ladder
        ladder = sharp_ladder if use_flat else flat_ladder
        idx = ladder_index(ladder, pc)
    if idx is None:
        # fallback, return original
        return pc
    new_idx = (idx + semitones) % 12
    return ladder[new_idx]

def chord_pitchclasses_abc(chord_sym):
    root, quality = chord_root_and_quality(chord_sym)
    if root is None:
        return set()
    root_pc = abc_pitchclass_for_root(root)
    steps = semitone_steps_for_quality(quality)
    pcs = set()
    for st in steps:
        pcs.add(transpose_pitchclass(root_pc, st))
    return pcs

def normalize_token_pitchclass(pitch_tok):
    if pitch_tok == "z":
        return "z"
    base = "".join(ch for ch in pitch_tok if ch not in ("'",","))
    return base


def generate_bar_pitches_with_chord(P, idx2tok, tok2idx,
                                    bar_note_count,
                                    prev_last_pitch_tok,
                                    chord_sym):
    START = "<START>"
    END = "<END>"

    real_idx = [i for i,t in enumerate(idx2tok) if t not in (START,END)]
    chord_pitchclasses = chord_pitchclasses_abc(chord_sym)

    def sample_from_row(curr_idx, allowed_indices):
        row = P[curr_idx]
        probs = np.zeros_like(row)
        probs[allowed_indices] = row[allowed_indices]
        s = probs.sum()
        if s <= 0:
            probs = np.zeros_like(row)
            probs[allowed_indices] = 1.0 / max(1,len(allowed_indices))
        else:
            probs /= s
        j = np.random.choice(np.arange(len(probs)), p=probs)
        return j

    # figure the starting Markov row for first note
    if prev_last_pitch_tok in tok2idx:
        curr_idx = tok2idx[prev_last_pitch_tok]
    else:
        curr_idx = tok2idx.get(START, 0)

    bar_notes = []
    chord_allowed_idx = []
    for k,tok in enumerate(idx2tok):
        if tok in (START,END):
            continue
        pc = normalize_token_pitchclass(tok)
        if pc == "z":
            continue
        if pc in chord_pitchclasses:
            chord_allowed_idx.append(k)

    if chord_allowed_idx:
        first_j = sample_from_row(curr_idx, chord_allowed_idx)
    else:
        first_j = sample_from_row(curr_idx, real_idx)

    first_tok = idx2tok[first_j]
    bar_notes.append(first_tok)
    curr_idx = first_j

    for _ in range(bar_note_count - 1):
        j = sample_from_row(curr_idx, real_idx)
        bar_notes.append(idx2tok[j])
        curr_idx = j

    return bar_notes


def generate_full_pitch_sequence(models, rhythm_bars,
                                 force_start="C", force_end="E"):

    chord_seq = generate_chord_sequence(models["chord"], len(rhythm_bars))

    Pp   = models["pitch"]["P"]
    pTok = models["pitch"]["idx2tok"]
    pMap = models["pitch"]["tok2idx"]

    bars_pitches = []
    prev_last = None
    for bar_i, bar_rhythm in enumerate(rhythm_bars):
        bar_len = len(bar_rhythm)  # number of notes in this bar
        bar_notes = generate_bar_pitches_with_chord(
            Pp, pTok, pMap,
            bar_len,
            prev_last,
            chord_seq[bar_i]
        )
        bars_pitches.append(bar_notes)
        prev_last = bar_notes[-1]


    flat_pitches = [pt for bar in bars_pitches for pt in bar]


    if force_start in pMap:
        flat_pitches[0] = force_start
    else:
        for tok in pTok:
            if tok in ("<START>","<END>"):
                continue
            stripped = tok.lstrip("^_=")
            if stripped and stripped[0] in ("C","c"):
                flat_pitches[0] = tok
                break

    if force_end in pMap:
        flat_pitches[-1] = force_end
    else:
        for tok in pTok[::-1]:
            if tok in ("<START>","<END>"):
                continue
            stripped = tok.lstrip("^_=")
            if stripped and stripped[0] in ("E","e"):
                flat_pitches[-1] = tok
                break

    return flat_pitches, chord_seq

def assemble_abc_from_pr(pitches,
                         rhythms,
                         meter,
                         chord_seq,
                         key="C",
                         base_len="1/8",
                         title="Generated Tune"):
    if meter == "3/4":
        per_bar = 6
    elif meter == "4/4":
        per_bar = 8
    else:
        raise ValueError("Meter must be '3/4' or '4/4'.")

    out_bars = []
    curr_bar_tokens = []
    curr_bar_idx = 0
    acc = 0

    if chord_seq:
        first_chord = chord_seq[curr_bar_idx]
        if first_chord:
            curr_bar_tokens.append(f"\"{first_chord}\"")

    for p, r in zip(pitches, rhythms):
        dur_val = int(r)
        token = p if r == "1" else f"{p}{r}"
        curr_bar_tokens.append(token)
        acc += dur_val

        if acc == per_bar:
            out_bars.append(" ".join(curr_bar_tokens))
            curr_bar_tokens = []
            acc = 0
            curr_bar_idx += 1

            if curr_bar_idx < len(chord_seq):
                next_chord = chord_seq[curr_bar_idx]
                if next_chord:
                    curr_bar_tokens.append(f"\"{next_chord}\"")

        elif acc > per_bar:
            raise RuntimeError("Rhythm overflow in bar assembly.")

    if curr_bar_tokens:
        out_bars.append(" ".join(curr_bar_tokens))

    header = [
        "X: 999",
        f"T: {title}",
        f"M: {meter}",
        f"L: {base_len}",
        f"K: {key}",
    ]

    body = " | ".join(out_bars)

    return "\n".join(header + [f"|: {body} :|"])



if __name__ == "__main__":
    np.random.seed(114514)

    models = train_models(ABC_TEXT)

    METER = "4/4"
    N_BARS = 8

    rhythm_bars, rhythm_flat = generate_rhythms(models["rhythm"], N_BARS, METER)

    pitch_flat, chord_seq = generate_full_pitch_sequence(
        models,
        rhythm_bars,
        force_start="C",
        force_end="E"
    )

    abc_out = assemble_abc_from_pr(
        pitch_flat,
        rhythm_flat,
        meter=METER,
        chord_seq=chord_seq,
        key="C",
        base_len="1/8",
        title=f"{N_BARS} bars in {METER}"
    )

    print("mel:")
    print(abc_out)
    print()