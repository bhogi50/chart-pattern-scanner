import pandas as pd

UP_COLOR = "#26a69a"
DOWN_COLOR = "#ef5350"
SUPPORT_COLOR = "#2196f3"
RESISTANCE_COLOR = "#ff9800"
PATTERN_LINE_COLOR = "#ab47bc"


def _d(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def candlestick_series(df):
    data=[]
    for idx,row in df.iterrows():
        data.append({"time":_d(idx),"open":round(float(row.Open),2),"high":round(float(row.High),2),"low":round(float(row.Low),2),"close":round(float(row.Close),2)})
    return {"type":"Candlestick","data":data,"options":{"upColor":UP_COLOR,"downColor":DOWN_COLOR,"borderVisible":False,"wickUpColor":UP_COLOR,"wickDownColor":DOWN_COLOR}}


def line_series(p1,p2,color,title,dashed=False):
    return {"type":"Line","data":[{"time":_d(p1[0]),"value":round(float(p1[1]),2)},{"time":_d(p2[0]),"value":round(float(p2[1]),2)}],"options":{"color":color,"lineWidth":2,"lineStyle":2 if dashed else 0,"title":title,"lastValueVisible":True,"priceLineVisible":False}}


def horizontal_line_series(price,start_date,end_date,color,title="",dashed=True):
    return line_series((start_date,price),(end_date,price),color,title,dashed)


def pattern_line_series(points,color=PATTERN_LINE_COLOR,title="Pattern"):
    if not points: return None
    return {"type":"Line","data":[{"time":_d(dt),"value":round(float(price),2)} for dt,price in points],"options":{"color":color,"lineWidth":2,"lineStyle":0,"title":title,"lastValueVisible":False,"priceLineVisible":False}}


def pattern_markers(points,direction):
    color={"Bullish":UP_COLOR,"Bearish":DOWN_COLOR,"Neutral":PATTERN_LINE_COLOR}.get(direction,PATTERN_LINE_COLOR)
    return [{"time":_d(dt),"position":"aboveBar" if i%2==0 else "belowBar","color":color,"shape":"circle","text":str(i+1)} for i,(dt,_) in enumerate(points)]


def base_chart_options(height=520):
    return {"height":height,"layout":{"background":{"type":"solid","color":"white"},"textColor":"#333"},"grid":{"vertLines":{"color":"#eee"},"horzLines":{"color":"#eee"}},"rightPriceScale":{"borderColor":"#ccc"},"timeScale":{"borderColor":"#ccc","timeVisible":False},"crosshair":{"mode":0}}


def build_chart_payload(df,support_levels=None,resistance_levels=None,pattern_match=None,height=520):
    series=[candlestick_series(df)]
    start,end=df.index[0],df.index[-1]
    for x in support_levels or []:
        series.append(horizontal_line_series(x["price"],start,end,SUPPORT_COLOR,f"S{x.get('rank','') or ''} {x['price']:.2f}"))
    for x in resistance_levels or []:
        series.append(horizontal_line_series(x["price"],start,end,RESISTANCE_COLOR,f"R{x.get('rank','') or ''} {x['price']:.2f}"))
    if pattern_match:
        if pattern_match.upper_line:
            series.append(line_series(pattern_match.upper_line[0],pattern_match.upper_line[1],RESISTANCE_COLOR,"Upper boundary"))
        if pattern_match.lower_line:
            series.append(line_series(pattern_match.lower_line[0],pattern_match.lower_line[1],SUPPORT_COLOR,"Lower boundary"))
        if pattern_match.points:
            p=pattern_line_series(pattern_match.points,title=pattern_match.name)
            if p: series.append(p)
        if pattern_match.neckline:
            series.append(line_series(pattern_match.neckline[0],pattern_match.neckline[1],"#616161","Neckline",True))
    if pattern_match and pattern_match.points:
        series[0]["markers"]=pattern_markers(pattern_match.points,pattern_match.direction)
    return [{"chart":base_chart_options(height),"series":series}]
