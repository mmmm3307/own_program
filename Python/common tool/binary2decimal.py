def decimal_to_binary(decimal_num):
    if decimal_num == 0:
        return "0"

    # 处理符号
    is_negative = decimal_num < 0
    decimal_num = abs(decimal_num)

    # 处理整数部分
    integer_part = int(decimal_num)
    fractional_part = decimal_num - integer_part

    binary_integer = ""
    while integer_part > 0:
        binary_integer = str(integer_part % 2) + binary_integer
        integer_part //= 2

    if binary_integer == "":
        binary_integer = "0"

    # 处理小数部分
    binary_fractional = ""
    precision = 10  # 控制小数点后的位数
    while fractional_part > 0 and len(binary_fractional) < precision:
        fractional_part *= 2
        bit = int(fractional_part)
        if bit == 1:
            fractional_part -= bit
            binary_fractional += "1"
        else:
            binary_fractional += "0"

    # 合并结果
    binary_result = binary_integer
    if binary_fractional:
        binary_result += "." + binary_fractional

    if is_negative:
        binary_result = "-" + binary_result

    return binary_result

def binary_to_decimal(binary_str):
    # 处理符号
    is_negative = binary_str[0] == "-"
    if is_negative:
        binary_str = binary_str[1:]

    # 分割整数部分和小数部分
    if "." in binary_str:
        integer_part_str, fractional_part_str = binary_str.split(".")
    else:
        integer_part_str = binary_str
        fractional_part_str = ""

    # 处理整数部分
    decimal_integer = 0
    for i, digit in enumerate(reversed(integer_part_str)):
        decimal_integer += int(digit) * (2 ** i)

    # 处理小数部分
    decimal_fractional = 0
    for i, digit in enumerate(fractional_part_str):
        decimal_fractional += int(digit) * (2 ** -(i + 1))

    decimal_result = decimal_integer + decimal_fractional

    if is_negative:
        decimal_result = -decimal_result

    return decimal_result

# decimal1 = 0.28125
# binary1 = decimal_to_binary(decimal1)
# print (f'decimal:{decimal1}--->binary:{binary1}')
binary2 = 0.0100010101
decimal2 = binary_to_decimal(binary2)
print (f'binary:{binary2}--->decimal:{decimal2}')