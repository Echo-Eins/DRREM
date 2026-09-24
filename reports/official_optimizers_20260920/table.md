| Вариант | Test h1 | Test среднее H | с/батч | байт/с | Пик МиБ | Состояние оптимизатора МиБ |
|---|---:|---:|---:|---:|---:|---:|
| Local gradient / level Adam | 4.3599 | 4.4615 | 0.896 | 1011 | 210.0 | 42.0 |
| Local gradient / level Muon | 4.3003 | 4.4058 | 0.946 | 986 | 192.0 | 24.0 |
| Local gradient / whole Muon | 4.3235 | 4.4087 | 0.966 | 947 | 192.0 | 24.0 |
| Full tick gradient / Adam | 4.1792 | 4.4148 | 1.291 | 724 | 210.5 | 42.0 |
| Full tick gradient / Muon | 4.2434 | 4.3802 | 1.297 | 707 | 192.5 | 24.0 |
| Finite contrast / official Adam | 4.1898 | 4.4214 | 0.895 | 1035 | 201.8 | 42.0 |
| Frozen core / local head objective | 4.6242 | 4.5553 | 0.891 | 1061 | 116.2 | 4.0 |
| Frozen core / full tick head objective | 4.6058 | 4.5389 | 0.899 | 1030 | 110.3 | 4.0 |
