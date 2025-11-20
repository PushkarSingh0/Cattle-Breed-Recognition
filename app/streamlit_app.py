import streamlit as st
import requests
from PIL import Image
import io

# --- Configuration ---
# This matches your app.py (host='0.0.0.0', port=5000)
SERVER_URL = "http://127.0.0.1:5000/predict" 
# --- End Configuration ---

st.set_page_config(
    page_title="Cattle Breed Recognizer",
    page_icon="🐄",
    layout="centered"
)

st.title("🐄 AI-Powered Cattle Breed Recognition")
st.markdown(
    """
    Upload an image of cattle or buffalo to classify its breed. 
    This application uses a two-stage model: 
    1.  A **Gatekeeper** to confirm it's cattle.
    2.  A **Classifier** to identify the specific breed.
    """
)

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload an image file (JPG, JPEG, or PNG) for classification."
)

if uploaded_file is not None:
    # --- Display Uploaded Image ---
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("---")

    # --- Prediction Button and Logic ---
    if st.button("Classify Breed"):
        
        # Show a spinner while processing
        with st.spinner("Classifying... Please wait."):
            
            # Convert image to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            file_bytes = img_bytes.getvalue()
            
            # Prepare the file for the POST request
            files = {'file': (uploaded_file.name, file_bytes, uploaded_file.type)}
            
            try:
                # --- Send Request to Server ---
                response = requests.post(SERVER_URL, files=files, timeout=60)
                
                # --- Handle Server Response ---
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("Prediction Results:")

                    # --- CASE 1: Gatekeeper rejected the image ---
                    if result.get("prediction_type") == "Invalid Image":
                        st.warning(f"**Gatekeeper:** {result.get('message', 'Image is not cattle or buffalo.')}")

                    # --- CASE 2: Gatekeeper passed, breed was classified ---
                    elif result.get("prediction_type") == "Breed":
                        st.info(f"**Animal Type:** {result.get('animal_type', 'N/A')}")
                        
                        # --- MODIFICATION: Handle list of predictions ---
                        predictions = result.get('predictions', [])
                        
                        if predictions:
                            # Display Top Prediction prominently
                            top_pred = predictions[0]
                            st.success(f"**Top Prediction:** {top_pred.get('breed', 'N/A')}")
                            st.metric(label="Confidence", value=top_pred.get('confidence', '0.00%'))

                            # Display Top 3 list
                            if len(predictions) > 1:
                                st.markdown("---")
                                st.subheader("Top 3 Results:")
                                # Use st.columns for a cleaner layout
                                cols = st.columns(3)
                                for i, pred in enumerate(predictions):
                                    with cols[i]:
                                        st.markdown(f"**{i+1}. {pred.get('breed')}**")
                                        st.markdown(f"{pred.get('confidence')}")
                        else:
                            st.error("Prediction failed: Server did not return a list of predictions.")
                        # --- End Modification ---
                        
                    # --- CASE 3: Unknown success response ---
                    else:
                        st.error("Received an unknown response from the server.")
                        st.json(result) # Show the raw JSON for debugging

                else:
                    st.error(f"Error from server (Code {response.status_code}):")
                    try:
                        st.json(response.json()) # Try to show server's error message
                    except:
                        st.text(response.text) # Fallback if error isn't JSON
            
            except requests.exceptions.ConnectionError:
                st.error(
                    "**Connection Error:** Could not connect to the prediction server. "
                    "Please ensure `app.py` is running in a terminal."
                )
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

else:
    st.info("Please upload an image to begin classification.")

st.sidebar.header("About")
st.sidebar.markdown(
    """
    This application is part of the **Smart India Hackathon 2025**. 
    It leverages a two-stage deep learning pipeline to identify cattle breeds from images.
    """
)