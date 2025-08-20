```python
def calculate_interest(principal, interest_rate, time_period, interest_type="simple", time_unit="years"):
    """Calculates simple or compound interest.

    Args:
        principal: The principal amount.
        interest_rate: The annual interest rate (decimal).
        time_period: The time period.
        interest_type: 'simple' or 'compound'. Defaults to 'simple'.
        time_unit: 'years', 'months', or 'days'. Defaults to 'years'.

    Returns:
        The calculated interest amount, or None if invalid input is provided.  
    """
    try:
        principal = float(principal)
        interest_rate = float(interest_rate)
        time_period = float(time_period)

        if principal < 0 or interest_rate < 0 or time_period < 0:
            return None

        if time_unit == "months":
            time_period /= 12
        elif time_unit == "days":
            time_period /= 365

        if interest_type == "simple":
            interest = principal * interest_rate * time_period
        elif interest_type == "compound":
            interest = principal * ( (1 + interest_rate)**time_period -1)
        else:
            return None

        return interest

    except ValueError:
        return None

if __name__ == "__main__":
    principal = input("Enter principal amount: ")
    interest_rate = input("Enter interest rate (decimal): ")
    time_period = input("Enter time period: ")
    time_unit = input("Enter time unit (years, months, days): ").lower()
    interest_type = input("Enter interest type (simple, compound): ").lower()

    interest = calculate_interest(principal, interest_rate, time_period, interest_type, time_unit)

    if interest is not None:
        print(f"Calculated interest: ${interest:.2f}")
    else:
        print("Invalid input provided.")

```