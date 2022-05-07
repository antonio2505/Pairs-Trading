
from alpaca_trade_api.rest import REST, TimeFrame
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import config


def main():

    #The mail addresses and password
    sender_address = config.EMAIL
    sender_pass = config.EMAIL_PWD
    receiver_address = config.RECEIVER_EMAIL
    #Setup the MIME
    message = MIMEMultipart()
    message['From'] = 'Trading Bot'
    message['To'] = receiver_address
    message['Subject'] = 'Pairs Trading Algo'   #The subject line

    #Selection of stocks
    #voir mon program notebook pour la Selection des paires de stock
    days = 300
    ADBE = 'ADBE'
    AAPL = 'AAPL'
    #Put Hisrorical Data into variables
    ADBE_df = api.get_bars("ADBE", TimeFrame.Hour, "2020-03-01", adjustment='raw').df
    ADBE_df.index = ADBE_df.index.tz_convert("America/New_York")
    ADBE_df = ADBE_df.between_time('09:31', '16:00')

    AAPL_df = api.get_bars(AAPL, TimeFrame.Hour,"2020-03-01", adjustment='raw').df
    AAPL_df.index = AAPL_df.index.tz_convert("America/New_York")
    AAPL_df = AAPL_df.between_time('09:31', '16:00')

    #Grab ADBE data and put in to a array
    data_1 = []
    times_1 = []
    for i in range(days):
        ADBE_close = ADBE_df.close[i]
        ADBE_time = ADBE_df.index[i]
        data_1.append(ADBE_close)
        times_1.append(ADBE_time)
    #Grab AAPL data and put in to an array
    data_2 = []
    times_2 = []
    for i in range(days):
        AAPL_close = AAPL_df.close[i]
        AAPL_time = AAPL_df.index[i]
        data_2.append(AAPL_close)
        times_2.append(AAPL_time)

    #les combiner
    hist_close = pd.DataFrame(data_1, columns=[ADBE])
    hist_close[AAPL] = data_2

    #dispersion(spread) entre les 2 stocks
    ADBE_curr = data_1[days-1]
    AAPL_curr = data_2[days-1]
    spread_curr = (ADBE_curr-AAPL_curr)

    #fixons une moyenne mobile(moving average) des 2 stocks
    move_avg_days = 5 # moyenne de 5 jours precedent

    #Moving average pour le ADBE
    ADBE_last = []
    for i in range(move_avg_days):
        ADBE_last.append(data_1[(days-1)-i])

    ADBE_hist = pd.DataFrame(ADBE_last)

    ADBE_mavg = ADBE_hist.mean()

    #Moving average for AAPL
    AAPL_last = []
    for i in range(move_avg_days):
        AAPL_last.append(data_2[(days-1)-i])
    AAPL_hist = pd.DataFrame(AAPL_last)
    AAPL_mavg = AAPL_hist.mean()

    #moyenne de la disperion(spread)
    spread_avg = min(ADBE_mavg - AAPL_mavg)

    #Spread_factor
    spreadFactor = .01 # determiner les l'ecart de la dispersion
    wideSpread = spread_avg*(1+spreadFactor)
    thinSpread = spread_avg*(1-spreadFactor)

    #Calc_of_shares_to_trade
    cash = float(account.buying_power)
    limit_ADBE = cash//ADBE_curr
    limit_AAPL = cash//AAPL_curr
    number_of_shares = int(min(limit_ADBE, limit_AAPL)/2)

    #Trading_algo
    #receuille les positon existante
    portfolio = api.list_positions()

    #receulle si le marcher est ouvert ou fermer
    clock = api.get_clock()

    #
    if clock.is_open == True: #si le marcher est ouvert
        if bool(portfolio) == False:

            #detect si la dispersion actuelle superieu que notre wideSpread fixer
            if spread_curr > wideSpread:
                #si tell est les cas alors signal de vente pour le 1er stock. et achat pour le deuxiem stock
                #d'apres l'equation si ADBE - AAPL > 0, alors ADBE > AAPL
                #
                api.submit_order(symbol = ADBE,qty = number_of_shares,side = 'sell',type = 'market',time_in_force ='day')

                #Long bottom stock(long AAPL)
                api.submit_order(symbol = AAPL,qty = number_of_shares,side = 'buy',type = 'market',time_in_force = 'day')

                mail_content = "Trades have been made, short top stock and long bottom stock"

            #on fait parail, mais de sign contraire maintenant
            elif spread_curr < thinSpread:
                #achat du ADBE
                api.submit_order(symbol = ADBE,qty = number_of_shares,side = 'buy',type = 'market',time_in_force = 'day')

                #vente AAPL
                api.submit_order(symbol = AAPL,qty = number_of_shares,side = 'sell',type = 'market',time_in_force ='day')
                mail_content = "Trades have been made, long top stock and short bottom stock"

        #cloturer notre position
        else:
            wideTradeSpread = spread_avg *(1+spreadFactor + .03)
            thinTradeSpread = spread_avg *(1+spreadFactor - .03)
            if spread_curr <= wideTradeSpread and spread_curr >=thinTradeSpread:
                api.close_position(ADBE)
                api.close_position(AAPL)
                mail_content = "Position has been closed"
            else:
                mail_content = "No trades were made, position remains open"
                pass
    else:
        mail_content = "The Market is Closed"

    #The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))
    #Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()

    done = 'Mail Sent'

    return done


if __name__=="__main__":

    api = REST(config.API_KEY, config.SECRET_KEY, base_url=config.BASE_URL)
    account = api.get_account()

    main()
