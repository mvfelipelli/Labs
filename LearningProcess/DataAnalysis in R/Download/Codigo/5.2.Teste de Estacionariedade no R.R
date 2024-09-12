#Series temporais e analises preditivas - Fernando Amaral

x = Box.test(airmiles,type="Ljung-Box")
print(x)

#faz a diferenciacao
z = diff(airmiles)

#testa novamente
x = Box.test(z,type="Ljung-Box")
print(x)

split.screen( figs = c( 2, 1 ) )
screen(1)
plot(airmiles, main="Airmiles")
screen(2)
plot(z, main="Série Transformada")
close.screen( all = TRUE )

#quantos processos de dif. precisam
#Phillips-Perron test
ndiffs(airmiles, test="pp")
ndiffs(z, test="pp")


















