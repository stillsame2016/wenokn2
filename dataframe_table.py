import uuid
import time
import pandas as pd
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from util import process_table_request


def render_interface_for_table(llm, llm2, index, pivot_table):
    buffered_table = st.session_state.wen_tables[index]
    with st.container():
        ''
        st.write(f"<div class='tableTitle'>Table {index +1}: {pivot_table.title}</div>", unsafe_allow_html=True)

        but_col1, but_col2, but_pad = st.columns([50 ,100 ,500])
        with but_col1:
            if st.button('Delete', key=f'delete-table-{index}'):
                del st.session_state.wen_datasets[index]
                del st.session_state.wen_tables[index]
                del st.session_state.table_chat_histories[index]
                del st.session_state.chart_types[index]
                st.session_state.rerun = True
                return

        with but_col2:
            if st.button('Change Chart Type', key=f'chart-type-{index}'):
                if st.session_state.chart_types[index] == 'bar_chart':
                    st.session_state.chart_types[index] = 'scatter_chart'
                elif st.session_state.chart_types[index] == 'scatter_chart':
                    st.session_state.chart_types[index] = 'line_chart'
                else:
                    st.session_state.chart_types[index] = 'bar_chart'


        with stylable_container(key=f'chart-panel-{index}', css_styles="padding-top: 100px;"):
            st.markdown("")
            if st.session_state.chart_types[index] == 'bar_chart':
                st.bar_chart(
                    buffered_table, # filtered_pivot_table,
                    # x='Date',
                    # y=pivot_table.variable_name,
                    x=buffered_table.columns[-2],
                    y=buffered_table.columns[-1],
                    color='Name',
                    height=450
                )
            elif st.session_state.chart_types[index] == 'scatter_chart':
                st.scatter_chart(
                    buffered_table, # filtered_pivot_table,
                    x=buffered_table.columns[-2],
                    y=buffered_table.columns[-1],
                    color='Name',
                    height=450
                )
            else:
                st.line_chart(
                    buffered_table, # filtered_pivot_table,
                    x=buffered_table.columns[-2],
                    y=buffered_table.columns[-1],
                    color='Name',
                    height=450
                )

        col3, pad, col4 = st.columns([30, 3, 20])
        with col3:
            st.dataframe(buffered_table, hide_index=True, use_container_width=True)
            with stylable_container(
                    key=f'add-to-map-{index}-button',
                    css_styles="""
                    button {
                        background-color: green;
                        color: white;
                        border-radius: 10px;
                        margin-left: 50px;
                    }
                    """,
            ):
                if not buffered_table['Name'].duplicated().any() and hasattr(pivot_table, 'use'):
                    if st.button('Add to Map', key=f'add-to-map-{index}'):
                        df = buffered_table.copy()
                        df.title = buffered_table.title
                        gdf = pivot_table.use.copy()

                        # Create a new column in df that matches the format of the Name column in gdf
                        df['CountyName'] = df['Name'].str.replace(' County', '')

                        # Perform the join operation
                        result = gdf.merge(df, left_on='Name', right_on='CountyName', how='left')

                        # Select the desired columns, including geometry and the original columns from df
                        result = result[['Name_y', 'geometry'] + df.columns[1:-1].to_list()]

                        # Optionally, rename 'Name_y' back to 'Name'
                        result = result.rename(columns={'Name_y': 'Name'})

                        result.attrs['data_name'] = df.title
                        result.label = df.title
                        result.id = str(uuid.uuid4())[:8]
                        result.time = time.time()

                        st.session_state.requests.append(df.title)
                        st.session_state.sparqls.append("Join")
                        st.session_state.datasets.append(result)
                        st.session_state.rerun = True

        with col4:
            table_chat_container = st.container(height=340)
            user_input_for_table = st.chat_input(f"What can I help you with Table {index+1}?")

            with table_chat_container:
                # Show the chat history
                for message in st.session_state.table_chat_histories[index]:
                    with st.chat_message(message['role']):
                        st.markdown(message['content'])

            if user_input_for_table:
                with table_chat_container:
                    st.chat_message("user").markdown(user_input_for_table)
                    st.session_state.table_chat_histories[index].append({"role": "user", "content": user_input_for_table})

                    response = process_table_request(llm, llm2, user_input_for_table, index)
                    if response["category"] == "Request data":
                        # st.code(response['answer'])
                        exec(response['answer'])
                        if isinstance(st.session_state.wen_tables[index], pd.Series):
                            st.session_state.wen_tables[index] = st.session_state.wen_tables[index].to_frame().T
                        st.session_state.wen_tables[index].title = response['title']
                        answer = f"""
                                    Your request has been processed. {st.session_state.wen_tables[index].shape[0]}
                                    { "rows are" if st.session_state.wen_tables[index].shape[0] > 1 else "row is"}
                                    found and displayed.
                                    """
                        st.chat_message("assistant").markdown(answer)
                        st.session_state.table_chat_histories[index].append({"role": "assistant", "content": answer})
                        st.session_state.rerun = True
                    else:
                        st.chat_message("assistant").markdown(response['answer'])
                        st.session_state.table_chat_histories[index].append \
                            ({"role": "assistant", "content": response['answer']})
