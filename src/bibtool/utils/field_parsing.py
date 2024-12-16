import warnings


def country_to_iso(country: str, ignore_error=False) -> str:
    # Check most common countries first
    if "China" in country or "Hunan" in country or "Macao" in country or "Hong Kong" in country:
        return "CHN"
    elif any(
        code in country
        for code in (
            "USA",
            "U.S.A",
            "United States",
            "Iowa",
            "North Carolina",
            "Massachusetts",
            "Oregon",
            "Illinois",
            "Maryland",
        )
    ):
        return "USA"
    # Then alphabetically ...
    elif country.endswith(("UAE", "United Arab Emirates")):
        return "ARE"
    elif country.endswith("Australia"):
        return "AUS"
    elif country.endswith("Austria"):
        return "AUT"
    elif country.endswith("Bangladesh"):
        return "BGD"
    elif country.endswith("Brazil"):
        return "BRA"
    elif country.endswith("Belgium"):
        return "BEL"
    elif country.endswith("Canada"):
        return "CAN"
    # China is check first for optimization
    elif country.endswith("Colombia"):
        return "COL"
    elif country.endswith("Switzerland"):
        return "CHE"
    elif country.endswith(("Czech Republic", "Czechia")):
        return "CZE"
    elif country.endswith("Germany"):
        return "DEU"
    elif country.endswith("Denmark"):
        return "DNK"
    elif country.endswith(("Algeria", "Algérie")):
        return "DZA"
    elif country.endswith("Estonia"):
        return "EST"
    elif country.endswith("Egypt"):
        return "EGY"
    elif country.endswith("Spain"):
        return "ESP"
    elif country.endswith("France"):
        return "FRA"
    elif country.endswith("Finland"):
        return "FIN"
    elif country.endswith(
        ("UK", "U.K", "United Kingdom", "England", "Scotland", "Wales", "Northern Ireland", "London")
    ):
        return "GBR"
    elif country.endswith("Greece"):
        return "GRC"
    elif country.endswith("India") or "West Bengal" in country:
        return "IND"
    elif country.endswith(("Iran", "Iran,", "Isfahan Iran")):
        return "IRN"
    elif country.endswith("Iraq"):
        return "IRQ"
    elif country.endswith("Iceland"):
        return "ISL"
    elif country.endswith("Israel"):
        return "ISR"
    elif country.endswith("Italy"):
        return "ITA"
    elif country.endswith("Japan"):
        return "JPN"
    elif country.endswith("Jordan"):
        return "JOR"
    elif any(term in country for term in ("Korea", "Seoul", "Jeonju", "Gwangju")):
        return "KOR"
    elif country.endswith("Lithuania"):
        return "LTU"
    elif country.endswith(("Mexico", "México", "DF")):
        return "MEX"
    elif country.endswith("Montenegro"):
        return "MNE"
    elif country.endswith("Malaysia"):
        return "MYS"
    elif country.endswith("Malawi"):
        return "MWI"
    elif country.endswith("Netherlands"):
        return "NLD"
    elif country.endswith("Nepal"):
        return "NPL"
    elif country.endswith("Norway"):
        return "NOR"
    elif country.endswith("Pakistan"):
        return "PAK"
    elif country.endswith("Peru"):
        return "PER"
    elif country.endswith("Poland"):
        return "POL"
    elif country.endswith("Portugal"):
        return "PRT"
    elif country.endswith("Russia"):
        return "RUS"
    elif country.endswith(("Saudi Arabia", "Saudia Arabia", "Arabia", "P. O. Box Saudi Arabia")):
        return "SAU"
    elif country.endswith("Sweden"):
        return "SWE"
    elif country.endswith("Singapore"):
        return "SGP"
    elif country.endswith("Tunisia"):
        return "TUN"
    elif country.endswith("Turkey"):
        return "TUR"
    # USA are check first for optimization
    elif country.endswith(("South Africa", "Durban South Africa")):
        return "ZAF"
    elif not ignore_error:
        warnings.warn(f"Unkown country: {country}", stacklevel=2)

    return ""


def date_to_year(date: str) -> int | None:
    import re

    year = re.match(r"\d{4}", date)

    return int(year.group(0).strip()) if year else None
