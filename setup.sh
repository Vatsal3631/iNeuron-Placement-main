mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"tikiwalavr@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[theme]\n\
primaryColor = '#F63366'\n\
backgroundColor = '#BFECE5'\n\
secondaryBackgroundColor = '#B3C3E2'\n\
textColor= '#121111'\n\
font = 'sans serif'\n\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
