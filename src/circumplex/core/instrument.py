from typing import Any, Dict

import pandas as pd


class Instrument:
    def __init__(
        self,
        scales: pd.DataFrame,
        anchors: pd.DataFrame,
        items: pd.DataFrame,
        norms: Dict[str, Any],
        details: Dict[str, Any],
    ):
        self.scales = scales
        self.anchors = anchors
        self.items = items
        self.norms = norms
        self.details = details

    def __str__(self):
        return (
            f"{self.details['Abbrev']}: {self.details['Name']}\n"
            f"{self.details['Items']} items, {self.details['Scales']} scales, "
            f"{len(self.norms[1])} normative data sets\n"
            f"{self.details['Reference']}\n"
            f"<{self.details['URL']}>"
        )

    def summary(self, scales=True, anchors=True, items=True, norms=True):
        output = [str(self)]
        if scales:
            output.append("\n" + self.get_scales())
        if anchors:
            output.append("\n" + self.get_anchors())
        if items:
            output.append("\n" + self.get_items())
        if norms:
            output.append("\n" + self.get_norms())
        return "\n".join(output)

    def get_scales(self, items=False):
        output = [
            f"The {self.details['Abbrev']} contains {self.details['Scales']} circumplex scales."
        ]
        for _, scale in self.scales.iterrows():
            output.append(
                f"{scale['Abbrev']}: {scale['Label']} ({scale['Angle']} degrees)"
            )
            if items:
                item_nums = [int(i) for i in scale["Items"].split(",")]
                for num in item_nums:
                    item = self.items.loc[self.items["Number"] == num, "Text"].iloc[0]
                    output.append(f"    {num}. {item}")
        return "\n".join(output)

    def get_items(self):
        output = [
            f"The {self.details['Abbrev']} contains {self.details['Items']} items ({self.details['Status']}):"
        ]
        for _, item in self.items.iterrows():
            if not pd.isna(item["Number"]):
                output.append(f"{item['Number']}. {item['Text']}")
            else:
                output.append(item["Text"])
        return "\n".join(output)

    def get_anchors(self):
        output = [
            f"The {self.details['Abbrev']} is rated using the following {len(self.anchors)}-point scale."
        ]
        for _, anchor in self.anchors.iterrows():
            output.append(f"{anchor['Value']}. {anchor['Label']}")
        return "\n".join(output)

    def get_norms(self):
        samples = self.norms[1]
        n_norms = len(samples)
        if n_norms == 0:
            return f"The {self.details['Abbrev']} currently has no normative data sets."

        output = [
            f"The {self.details['Abbrev']} currently has {n_norms} normative data set(s):"
        ]
        for i, sample in samples.iterrows():
            output.extend(
                [
                    f"{sample['Sample']}. {sample['Size']} {sample['Population']}",
                    sample["Reference"],
                    f"<{sample['URL']}>",
                ]
            )
        return "\n".join(output)


def instruments():
    return """The circumplex package currently includes 13 instruments:
1. CSIE: Circumplex Scales of Interpersonal Efficacy (csie)
2. CSIG: Circumplex Scales of Intergroup Goals (csig)
3. CSIP: Circumplex Scales of Interpersonal Problems (csip)
4. CSIV: Circumplex Scales of Interpersonal Values (csiv)
5. IGI-CR: Interpersonal Goals Inventory for Children, Revised Version (igicr)
6. IIP-32: Inventory of Interpersonal Problems, Brief Version (iip32)
7. IIP-64: Inventory of Interpersonal Problems (iip64)
8. IIP-SC: Inventory of Interpersonal Problems, Short Circumplex (iipsc)
9. IIS-32: Inventory of Interpersonal Strengths, Brief Version (iis32)
10. IIS-64: Inventory of Interpersonal Strengths (iis64)
11. IIT-C: Inventory of Influence Tactics Circumplex (iitc)
12. IPIP-IPC: IPIP Interpersonal Circumplex (ipipipc)
13. ISC: Interpersonal Sensitivities Circumplex (isc)"""


def instrument(code: str):
    # This function would load the instrument data from a file or database
    # For now, we'll just return a placeholder
    return f"Instrument data for {code}"
