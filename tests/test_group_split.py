dados pequenos feitos à mão: ~10 linhas, 4 grupos de tamanhos diferentes

teste_sem_vazamento:
    grupos do treino e grupos do teste não têm elementos em comum

teste_cobre_tudo:
    linhas do treino + linhas do teste = total, sem repetição

teste_reprodutivel:
    mesma semente -> mesmo resultado; sementes diferentes -> pode mudar

teste_erros:
    tamanhos diferentes de X, y e groups levantam ValueError
    test_size fora de (0, 1) levanta ValueError