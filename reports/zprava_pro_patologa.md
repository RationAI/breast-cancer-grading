# Zajímavé preparáty — model Nottinghamského gradingu

**Zpráva pro patologa · 12. 11. 2026**

## O modelu

Deep-learning model (multiple-instance learning, MIL) odhaduje **Nottinghamský grading**
na celých slidech. Model pracuje se 4 třídami:

| Značka | Význam |
|---|---|
| **N** | bez tumorového epitelu |
| **G1** | grade 1 |
| **G2** | grade 2 |
| **G3** | grade 3 |

Na testovací sadě **266 preparátů** (nikdy je model při tréninku neviděl) určí správnou
třídu u **85,3 %** preparátů. Pozor na tohle číslo: sada obsahuje **131 preparátů bez tumoru**,
které model téměř vždy pozná (99 %), a ty ho vytahují nahoru. Na **135 preparátech s tumorem**
určí správný grade v **72 %** — po třídách: **G1 50 %, G2 74 %, G3 75 %**. Chyby jsou
téměř výhradně mezi sousedními grades. (V interních materiálech projektu se uvádí i „balanced
accuracy“ 0,74 — průměr úspěšnosti po jednotlivých třídách, kde vzácná třída G1 má stejnou
váhu jako častá třída N; proto se jinde můžete setkat s nižším číslem.) Na každém preparátu
model vrací pravděpodobnosti všech čtyř tříd.

## Jak číst overlay v prohlížeči xOpat

Každý preparát má barevnou vrstvu přes celý snímek:

- **černá** = model se na to místo **nedíval** (žádná pozornost),
- **čím světlejší, tím víc pozornosti** — bílá je maximum pro daný preparát,
- vrstva je poloprůhledná, takže tkáň pod ní zůstává vidět.

> ⚠️ **Důležité:** overlay ukazuje, **kam se model díval**, ne kde je nádor.
> Rozlišení nádor × stroma u tohoto modelu selhává (AUC 0,48, tj. úroveň náhody).
> Pozornost často souvisí s morfologií, kterou grading naučen byl — dokážeme toto
> prohlédnout společně právě na preparátech níže.

## 1. Největší chyby modelu (skok o ≥ 1 třídu)

Sestupně podle velikosti chyby; „jistota" = pravděpodobnost, kterou model dal
**správné** třídě.

| Preparát | Pravda | Model | Jistota správné třídy | Poznámka |
|---|---|---|---|---|
| `2021_00353-21-T2` | **G3** | N | 7 % | Jediný G3 označený za „bez tumoru". Stojí za revizi, zda v digestu vůbec je tumorový epitel. |
| `2021_01085-02-N` | **N** | G3 | 34 % | Opak: negativ nazván G3. Navíc **extrémně zaměřená pozornost** (viz §3) — model „našel" malé místo a podle něj rozhodl. |
| `2021_02976-08-T1` | **G1** | G3 | 0,2 % | Největší přeřazení nahoru (G1→G3, model 88 % na G3). Klinicky nejtěžší typ chyby. |
| `2021_06916-02-T` | **G1** | G3 | 0,5 % | Opět G1→G3 (74 % na G3). |
| `2024_00279-2-T` | **G2** | G3 | 6 % | Přeřazení o stupeň nahoru, 93 % na G3. |
| `2020_00168-02-T` | **G3** | G2 | 6 % | Podceňující chyba; velký preparát (17 tis. dlaždic), pozornost rozptýlená. |
| `2024_01350-1-T` | **G2** | G1 | 7 % | Podcenění o stupeň (92 % na G1). |
| `2024_05767-2-T` | **G3** | G2 | 11 % | Podcenění; pozornost zaměřená na ~500 dlaždic z 1 624. |

## 2. Nejistá (hraniční) rozhodnutí

Rozdíl dvou nejvyšších pravděpodobností menší než ~6 % — model si prakticky
„házet mincí" mezi dvěma třídami:

| Preparát | Pravda | Model | Rozdíl | Jistota |
|---|---|---|---|---|
| `2022_00840-1-T` | G2 | G1 | 1,8 % | 51 % : 49 % |
| `2023_09609-2-T` | G3 | G2 | 2,0 % | 50 % : 48 % |
| `2020_04844-02-T` | G3 | G2 | 5,5 % | 52 % : 46 % |
| `2021_07429-01-T` | G1 | G2 | 5,8 % | 51 % : 45 % |
| `2022_01879-11-T` | G2 | G3 | 6,3 % | 53 % : 47 % |

`2023_04624-1-T` je jediný správně klasifikovaný preparát s takto malou jistotou
(G2, 52 %) — vhodný příklad „správně, ale náhodou".

## 3. Mimořádně zaměřená pozornost

Model rozhodl podle malého ostrůvku tkáně (efektivní počet dlaždic pozornosti
`eff` = kolika dlaždicemi z celého preparátu se model reálně „díval"):

| Preparát | Pravda | Model | eff / celkem | Co je zajímavé |
|---|---|---|---|---|
| `2021_01085-02-N` | N | **G3 ✗** | **173 / 6 316** | Nejzaměřenější chybný preparát — chyba + extrémní fokus v jednom. |
| `2024_02939-1-N` | N | N ✓ | 137 / 3 830 | Správně, ale rozhodl z 4 % preparátu. |
| `2021_00646-01-N` | N | N ✓ (46 %) | 185 / 4 205 | Správně, ale nejistě — pozornost na malém místě. |
| `2021_05142-01-N` | N | N ✓ | 127 / 2 387 | Správně, fokus na ~5 % preparátu. |
| `2023_03111-1-N` | N | N ✓ | 141 / 2 237 | Správně, fokus na ~6 % preparátu. |
| `2022_04814-2-T` | G2 | G1 ✗ | 152 / 2 371 | Chyba u silně zaměřeného tumoru. |

U správně klasifikovaných N preparátů stojí za to zjistit, **na co** se model
díval — jestli to je např. běžný epitel, nebo něco, co vypadá jako grading-
-relevantní morfologie.

## 4. Zvláštní případ

- `2021_07429-02-N` — **49 dlaždic** na celém preparátu (ostatní mají tisíce).
  Model ho klasifikoval správně (N, 100 %), ale jde o degradovaný/neúplný
  materiál; doporučuji ověřit kvalitu skenu před jakoukoliv interpretací.

## Matice záměn (266 testovacích preparátů)

Řádky = pravda, sloupce = predikce modelu.

| | N | G1 | G2 | G3 |
|---|---|---|---|---|
| **N** | **130** | 0 | 0 | 1 |
| **G1** | 0 | **7** | 5 | 2 |
| **G2** | 0 | 9 | **43** | 6 |
| **G3** | 1 | 0 | 15 | **47** |

Přehledně po třídách: **N 130/131 (99 %), G1 7/14 (50 %), G2 43/58 (74 %), G3 47/63 (75 %)**.
Chyby se téměř výhradně dějí **mezi sousedními grades** (G2↔G1, G2↔G3); skok přes dva grades
je vzácný (4 případy výše v §1). Podhodnocení (model dal nižší grade, než je pravda):
**25 preparátů (9,4 %)**, nadhodnocení: 14 preparátů.

## Kde preparáty najít

- Interaktivní soubor se všemi 266 preparáty (klikatelné odkazy do xOpat):
  **MLflow → experiment 82 „BRC Grading" → run `c9c975a3` → Artifacts → `test_grey.html`**
  (stejný soubor i na clusteru: `/mnt/projects/breast_cancer/brc_grading/reports/test_grey.html`)
- Tabulka se všemi čísly pro všechny 266 preparátů: `summary.csv` ve stejném
  adresáři (`masks/test/summary.csv` na disku uvedeném výše).
