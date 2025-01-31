import streamlit as st
import pandas as pd
import psycopg2

# PostgreSQL database configuration
DB_HOST = 'localhost'
DB_NAME = 'NewDB'
DB_USER = 'postgres'
DB_PASS = '123'

def get_data_from_db(query, params=None):
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df

def execute_query(query, data):
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    cursor = conn.cursor()
    cursor.execute(query, data)
    conn.commit()
    cursor.close()
    conn.close()

# Check if the user is logged in
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning('Please log in to access this page.')
    st.stop()

# Set the title of the Streamlit app
st.title('Community Posts for Farmers')

# Display the form to submit a new post
st.subheader('Submit a New Post')
with st.form(key='post_form'):
    title = st.text_input('Title')
    body = st.text_area('Post Body')
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        if st.session_state['username']:
            user_query = "SELECT user_id FROM users WHERE username = %s"
            user_df = get_data_from_db(user_query, (st.session_state['username'],))
            
            if not user_df.empty:
                user_id = int(user_df.iloc[0]['user_id'])  # Ensure user_id is of type int
                insert_query = """
                INSERT INTO posts (user_id, title, body)
                VALUES (%s, %s, %s)
                """
                execute_query(insert_query, (user_id, title, body))
                st.success('Post submitted successfully!')
            else:
                st.error('User not found.')
        else:
            st.error('User not logged in.')

# Display all posts
st.subheader('All Posts')

posts_query = """
SELECT p.post_id, u.username, p.title, p.body, p.created_at
FROM posts p
JOIN users u ON p.user_id = u.user_id
ORDER BY p.created_at DESC
"""
posts_df = get_data_from_db(posts_query)

for index, post in posts_df.iterrows():
    with st.expander(f"Post ID: {post['post_id']} - {post['title']} by {post['username']}"):
        st.markdown(f"**{post['title']}**")
        st.markdown(f"*Posted by {post['username']} on {post['created_at']}*")
        st.write(post['body'])
        
        comments_query = """
        SELECT c.comment_id, u.username, c.comment_body, c.created_at
        FROM comments c
        JOIN users u ON c.user_id = u.user_id
        WHERE c.post_id = %s
        ORDER BY c.created_at ASC
        """
        comments_df = get_data_from_db(comments_query, (post['post_id'],))
        
        if not comments_df.empty:
            st.subheader('Comments')
            for _, comment in comments_df.iterrows():
                st.markdown(f"**{comment['username']}** commented on {comment['created_at']}")
                st.write(comment['comment_body'])
                st.markdown("---")
        else:
            st.write("No comments yet.")

        # Form to submit a new comment
        st.subheader('Submit a Comment')
        with st.form(key=f'comment_form_{post["post_id"]}'):
            comment_body = st.text_area('Comment Body', key=f'body_{post["post_id"]}')
            comment_submit_button = st.form_submit_button(label='Submit Comment')

            if comment_submit_button:
                if st.session_state['username']:
                    user_query = "SELECT user_id FROM users WHERE username = %s"
                    user_df = get_data_from_db(user_query, (st.session_state['username'],))
                    
                    if not user_df.empty:
                        user_id = int(user_df.iloc[0]['user_id'])  # Ensure user_id is of type int
                        comment_insert_query = """
                        INSERT INTO comments (post_id, user_id, comment_body)
                        VALUES (%s, %s, %s)
                        """
                        execute_query(comment_insert_query, (post['post_id'], user_id, comment_body))
                        st.success('Comment submitted successfully!')
                    else:
                        st.error('User not found.')
                else:
                    st.error('User not logged in.')
