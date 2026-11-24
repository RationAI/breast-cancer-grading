# Shoda mezi modelem MammaPrint a naším modelem hodnocení Grade 3

**Kontext.** Porovnali jsme dva modely na 304 snímcích (stejná skenovaná data,
stejná mřížka úrysků, překryv 100 %):

1. **MammaPrint** – externí model předpovídající rizikový index (model
   p. kolegy; možný průnik trénovacích dat jsme vyloučili – svůj model jsme na
   datech MammaPrint netrénovali),
2. **náš model** – klasifikátor Grade 3 s attention mapou (kde model „vidí“
   tumor).

Ptali jsme se dvou věcí: **(a)** dívá se náš model na stejné části tkáně jako
MammaPrint? **(b)** korelují jejich výsledky navzájem?

## (a) Které úrysky jsou důležité — shoda prakticky žádná

Pro každý snímek jsme spočítali Spearmanovu korelaci ρ mezi attention našího
modelu a „řidiči rizika“ MammaPrintu (úrysky, jejichž vynecháním rizikový
index klesne). ρ nabývá hodnot −1 až +1: **0 = žádná souvislost (hod mincí)**,
+1 = oba modely seřadí úrysky úplně stejně, −1 = přesně naopak; |ρ| < 0,2
považujeme za shodu nulovou.

| Srovnání (304 snímků, medián ρ) | Hodnota | Výklad |
|---|---|---|
| naše attention vs. **řidiči rizika MammaPrint** | **+0,02** (52 % snímků kladných) | mince – žádná shoda |
| naše attention vs. velikost jeho vlivu (|LOO|) | +0,01 | nic |
| naše attention vs. jeho **hrubá attention** | **+0,62** (96 % snímků kladných) | oba modely upřednostňují tumor před pozadím |

Jinými slovy: **oba modely se dívají na tumor, ale který tumor „dělá riziko“,
náš model z grade attention nepozná.** Zelená vrstva v prohlížeči (průnik
obou horních decilů) to potvrzuje vizuálně.

## (b) Výsledkové skóre — naopak nepřímá vazba

| Srovnání na úrovni snímku (304 snímků) | Spearman ρ |
|---|---|
| sanity check: jeho index vs. skutečný index | +0,71 ✓ |
| **jeho rizikový index vs. naše P(Grade 3)** | **−0,72** |

Kde MammaPrint vidí vysoké riziko, náš model vidí spíše *nižší* grade. To
odpovídá klinice: MammaPrint je na grade nezávislý (proliferační signál),
grade je naopak o morfologii. Systémy se tedy **neduplikují – měří různé
vlastnosti téže barveniny.**

## Závěr

> Oba modely vyhledávají tumor se shodnou preferencí (ρ 0,62), ale které
> partie tumoru dělají vysoké MammaPrint riziko, je pro naši grade attention
> neviditelné (ρ 0,02). Skóre si neodporují – jen měří různé vlastnosti téže
> barveniny.

## Interaktivní prohlídka (304 karet)

Každá karta otevírá snímek v xOpat prohlížeči (vyžaduje firemní VPN); čísla
na kartách jsou platná i offline. Vrstvy:

- **zelená** (zapnutá) – shoda: úrysky, které oba modely řadí do své horní
  desítky,
- **oranžová** – řidiče rizika MammaPrint (vynechání sníží jeho index),
- **šedá** – naše attention: **čím světlejší, tím vyšší** pozornost
  (bílá = 100× průměrného úrysku); slabá = na snímku nic nevyčnívá.

Čísla na kartě: `P(G0-3)` = naše predikované pravděpodobnosti jednotlivých
stupňů grade (G3 = nejvyšší), `ρ order-vs-drivers` = jak dobře naše attention
umí seřadit stejné úrysky jako řidiči rizika (škála −1…+1 vysvětlená výše),
`ρ order-vs-his-attention` stejně proti jeho hrubé attention.

*Podrobnosti a všechny statistiky: `mammaprint_agreement_summary.md`,
`mammaprint_compare/report_summary.csv`; soubor `mammaprint_agreement_v3.html`.*
