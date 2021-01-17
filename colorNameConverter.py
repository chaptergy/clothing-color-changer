colorNames = [[[0, 0, 0], "Schwarz"], [[60, 61, 188], "Dunkles Schiefergrau"], [[105, 57, 144], "Schiefergrau"],
              [[105, 57, 153], "Helles Schiefergrau"], [[107, 53, 222], "Helles Stahlblau"],
              [[0, 0, 105], "Mattes Grau"], [[0, 0, 128], "Grau"], [[0, 0, 169], "Dunkelgrau"], [[0, 0, 192], "Silber"],
              [[0, 0, 211], "Hellgrau"], [[0, 0, 220], "Gainsboro"], [[0, 0, 245], "Rauchiges Weiß"],
              [[120, 7, 255], "Geisterweiß"], [[0, 0, 255], "Weiß"], [[0, 5, 255], "Schneeweiß"],
              [[30, 15, 255], "Elfenbein"], [[20, 15, 255], "Blütenweiß"], [[12, 17, 255], "Muschel"],
              [[20, 23, 253], "Altgold"], [[15, 20, 250], "Leinenfarbe"], [[17, 36, 250], "Antikes Weiß"],
              [[18, 50, 255], "Mandelweiß"], [[19, 42, 255], "Cremiges Papaya"], [[30, 26, 245], "Beige"],
              [[24, 35, 255], "Mais"], [[30, 41, 250], "Helles Goldrutengelb"], [[30, 31, 255], "Hellgelb"],
              [[27, 50, 255], "Chiffongelb"], [[27, 73, 238], "Blasse Goldrutenfarbe"], [[27, 106, 240], "Khaki"],
              [[30, 255, 255], "Gelb"], [[25, 255, 255], "Gold"], [[19, 255, 255], "Orange"],
              [[16, 255, 255], "Dunkles Orange"], [[21, 218, 218], "Goldrute"],
              [[21, 240, 184], "dunkle Goldrutenfarbe"], [[15, 177, 205], "Peru"], [[12, 219, 210], "Schokolade"],
              [[12, 220, 139], "Sattelbraun"], [[10, 183, 160], "Ocker"], [[0, 190, 165], "Braun"],
              [[0, 255, 139], "Dunkelrot"], [[0, 255, 128], "Kastanienbraun"], [[0, 206, 178], "Ziegelfarbe"],
              [[0, 141, 205], "Indischrot"], [[174, 232, 220], "Karmesinrot"], [[0, 255, 255], "Rot"],
              [[8, 255, 255], "Orangenrot"], [[5, 184, 255], "Tomatenrot"], [[8, 175, 255], "Koralle"],
              [[3, 139, 250], "Lachs"], [[0, 119, 240], "Helles Korallenrot"], [[8, 121, 233], "Dunkle Lachsfarbe"],
              [[9, 133, 255], "Helle Lachsfarbe"], [[14, 155, 244], "Sandbraun"], [[0, 61, 188], "Rosiges Braun"],
              [[17, 85, 210], "Gelbbraun"], [[17, 100, 222], "Grobes Braun"], [[20, 69, 245], "Weizen"],
              [[14, 70, 255], "Pfirsich"], [[18, 82, 255], "Navajoweiß"], [[16, 59, 255], "Tomatencreme"],
              [[170, 15, 255], "Rosige Lavenderfarbe"], [[3, 30, 255], "Altrosa"], [[175, 63, 255], "Rosa"],
              [[175, 73, 255], "Hellrosa"], [[165, 150, 255], "Leuchtendes Rosa"], [[150, 255, 255], "Fuchsie"],
              [[150, 255, 255], "Magentarot"], [[164, 235, 255], "Tiefrosa"], [[161, 228, 199], "Mittleres Violettrot"],
              [[170, 125, 219], "Blasses Violettrot"], [[150, 70, 221], "Pflaume"], [[150, 30, 216], "Distel"],
              [[120, 20, 250], "Lavendelfarbe"], [[150, 116, 238], "Violett"], [[151, 124, 218], "Orchidee"],
              [[150, 255, 139], "Dunkles Magentarot"], [[150, 255, 128], "Violett"], [[137, 255, 130], "Indigo"],
              [[136, 206, 226], "Blauviolett"], [[141, 255, 211], "Dunkles Violett"],
              [[140, 193, 204], "Dunkle Orchideenfarbe"], [[130, 125, 219], "Mittleres Violett"],
              [[144, 152, 211], "Mittlere Orchideenfarbe"], [[124, 144, 238], "Mittleres Schieferblau"],
              [[124, 143, 205], "Schieferblau"], [[124, 143, 139], "Dunkles Schieferblau"],
              [[120, 198, 112], "Mitternachtsblau"], [[120, 255, 128], "Marineblau"], [[120, 255, 139], "Dunkelblau"],
              [[120, 255, 205], "Mittelblau"], [[120, 255, 255], "Blau"], [[112, 181, 225], "Königsblau"],
              [[104, 156, 180], "Stahlblau"], [[109, 147, 237], "Kornblumenblau"], [[105, 225, 255], "Dodger-Blau"],
              [[98, 255, 255], "Tiefes Himmelblau"], [[101, 117, 250], "Helles Himmelblau"],
              [[99, 109, 235], "Himmelblau"], [[97, 63, 230], "Hellblau"], [[90, 255, 255], "Zyanblau"],
              [[90, 255, 255], "Blaugrün"], [[93, 60, 230], "Taubenblau"], [[90, 31, 255], "Helles Cyanblau"],
              [[37, 255, 206], "Aliceblau"], [[90, 15, 255], "Himmelblau"], [[75, 10, 255], "Cremig Pfefferminz"],
              [[60, 15, 255], "Honigmelone"], [[80, 128, 255], "Aquamarinblau"], [[87, 182, 224], "Türkis"],
              [[90, 68, 238], "Blasses Türkis"], [[89, 167, 209], "Mittleres Türkis"],
              [[90, 255, 209], "Dunkles Türkis"], [[80, 128, 205], "Mittleres Aquamarinblau"],
              [[88, 209, 178], "Helles Seegrün"], [[90, 255, 139], "Dunkles Zyanblau"], [[90, 255, 128], "Entenbraun"],
              [[91, 104, 160], "Kadettblau"], [[73, 170, 179], "Mittleres Seegrün"], [[60, 61, 188], "Dunkles Seegrün"],
              [[60, 101, 238], "Hellgrün"], [[60, 101, 251], "Blassgrün"], [[78, 255, 250], "Mittleres Frühlingsgrün"],
              [[75, 255, 255], "Frühlingsgrün"], [[60, 255, 255], "Zitronengrün"], [[60, 193, 205], "Gelbgrün"],
              [[73, 171, 139], "Seegrün"], [[60, 193, 139], "Waldgrün"], [[60, 255, 128], "Grün"],
              [[60, 255, 100], "Dunkelgrün"], [[40, 192, 142], "Olivfarbiges Graubraun"],
              [[41, 143, 107], "Dunkles Olivgrün"], [[30, 255, 128], "Olivgrün"], [[28, 111, 189], "Dunkles Khaki"],
              [[40, 193, 205], "Gelbgrün"], [[45, 255, 255], "Hellgrün"], [[42, 208, 255], "Grüngelb"]]


def hsv_to_name(hsv):
    """
    Converts a HSV value into a color name
    :param hsv: The hsv value [0..180, 0..255, 0..255]
    """
    min_colours = {}
    for color in colorNames:
        h_c, s_c, v_c = color[0]
        hd = abs(h_c - hsv[0])
        sd = abs(s_c - hsv[1])
        vd = abs(v_c - hsv[2])
        min_colours[(hd + sd + vd)] = color
    return min_colours[min(min_colours.keys())][1]
