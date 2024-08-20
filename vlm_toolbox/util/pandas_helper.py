

def show_df(df, head=True, cnt=5, logging_fn=print):
    logging_fn(f'shape: {df.shape}')
    if head:
        return df.head(cnt)
