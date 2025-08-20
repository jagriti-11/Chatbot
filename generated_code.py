```python
import re

# Pen price (can be modified easily)
PEN_PRICE = 2.50

def calculate_pens_sold(total_money_str):
    """Calculates the number of pens sold based on total money received."""

    try:
        # Clean up currency symbols and commas
        total_money_str = re.sub(r'[€$£,]', '', total_money_str)
        total_money = float(total_money_str)

        if total_money <= 0:
            raise ValueError("Total money received must be positive.")

        if PEN_PRICE == 0:
            raise ZeroDivisionError("Pen price cannot be zero.")

        pens_sold = total_money / PEN_PRICE
        return int(pens_sold)  # Round down to nearest whole number

    except ValueError as e:
        print(f"Error: Invalid input. {e}")
        return None
    except ZeroDivisionError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    total_money_input = input("Enter the total amount of money received from pen sales: ")
    pens_sold = calculate_pens_sold(total_money_input)

    if pens_sold is not None:
        print(f"Total number of pens sold: {pens_sold}")

```