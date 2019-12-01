cute_0 = 0b0000000001000000
cute_0 = ((cute_0 >> 1) & 0x5555) | ((cute_0 << 1) & 0xAAAA)
print(bin(cute_0))
cute_0 = ((cute_0 >> 2) & 0x3333) | ((cute_0 << 2) & 0xCCCC)
print(bin(cute_0))
cute_0 = ((cute_0 >> 4) & 0X0F0F) | ((cute_0 << 4) & 0xF0F0)
print(bin(cute_0))
cute_0 = (cute_0 >> 8) | (cute_0 << 8)
print(bin(cute_0))
