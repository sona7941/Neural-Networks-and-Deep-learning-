# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io
import os

st.set_page_config(page_title="LSTM Models", layout="wide")
st.title("🧠 LSTM Models: Next-Word & Hit Song Predictor")

# -------------------
# Utility functions
# -------------------
def preprocess_text(text):
    """Clean and preprocess text data"""
    text = str(text).lower()
    # Remove special characters but keep periods and common punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def sample_with_temperature(preds, temperature=1.0):
    """Sample next word with temperature control"""
    preds = np.asarray(preds).astype('float64')
    if temperature <= 0:
        return np.argmax(preds)
    preds = np.log(preds + 1e-12) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def top_k_from_preds(preds, k=5):
    """Get top k predictions with probabilities"""
    idxs = np.argsort(preds)[-k:][::-1]
    return idxs, preds[idxs]

# -------------------
# Program 1: Next-Word Prediction using LSTM
# -------------------
st.sidebar.header("📝 Program 1: Next-Word Prediction")

# Dataset suggestions
with st.sidebar.expander("📊 Recommended Datasets"):
    st.write("""
    **Best Datasets for Next-Word Prediction:**
    1. **Reddit Comments Dataset** - Natural conversational text
    2. **OpenWebText** - Large corpus of web text
    3. **Cornell Movie Dialogues** - Conversational data
    4. **Customer Support Conversations** - Domain-specific
    5. **Simple Wikipedia** - Clean, simple language
    
    **Quick Test Options:**
    - Upload any .txt file with conversational text
    - Use customer service chat logs
    - News articles or book excerpts
    """)

uploaded_text = st.sidebar.file_uploader("Upload corpus (.txt or .csv)", type=["txt", "csv"])
use_sample_data = st.sidebar.checkbox("Use sample customer support data for testing")

df_text = None
corpus_text = None

# Sample data for testing
if use_sample_data:
    sample_conversations = [
        "i am not able to login to my account",
        "i cannot access my dashboard",
        "i am having trouble with my password",
        "i need help with my billing information",
        "i want to cancel my subscription",
        "i am experiencing technical difficulties",
        "i cannot find my order history",
        "i need assistance with my profile",
        "i am unable to download the app",
        "i want to update my payment method",
        "i have questions about my invoice",
        "i need to reset my password",
        "i cannot log into the system",
        "i am having issues with the website",
        "i need help troubleshooting the problem"
    ]
    corpus_text = ' . '.join(sample_conversations).lower()
    st.info("Using sample customer support conversations for testing")

if uploaded_text is not None:
    fname = uploaded_text.name.lower()
    if fname.endswith(".txt"):
        raw = uploaded_text.read()
        try:
            corpus_text = raw.decode('utf-8')
        except Exception:
            corpus_text = raw.decode('latin-1')
        corpus_text = preprocess_text(corpus_text)
        st.header("📝 Next-Word Prediction (Text File)")
        st.write(f"Loaded text file: {uploaded_text.name}")
    else:
        df_text = pd.read_csv(uploaded_text)
        st.header("📝 Next-Word Prediction (CSV)")
        st.write("Dataset preview:")
        st.dataframe(df_text.head())

# If CSV was loaded, let user pick text column
if df_text is not None:
    text_cols = df_text.select_dtypes(include=['object']).columns.tolist()
    if not text_cols:
        st.error("❌ No text columns detected in the uploaded CSV.")
    else:
        text_col = st.sidebar.selectbox("Choose text column", text_cols, index=0)
        # Sample data if too large
        max_rows = st.sidebar.number_input("Max rows to process (0 = all)", min_value=0, value=1000, step=100)
        if max_rows > 0 and len(df_text) > max_rows:
            df_text = df_text.head(max_rows)
            st.warning(f"⚠️ Using first {max_rows} rows for processing")
        
        # Join rows into corpus
        corpus_text = df_text[text_col].astype(str).apply(preprocess_text).str.cat(sep=' . ')

# Only proceed if corpus_text is available
if corpus_text:
    st.write(f"📊 Corpus length: {len(corpus_text)} characters")
    st.write(f"📊 Word count: {len(corpus_text.split())} words")
    
    # Show sample text
    with st.expander("👁️ Preview corpus text"):
        st.text(corpus_text[:500] + "..." if len(corpus_text) > 500 else corpus_text)

    # Model parameters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        max_vocab = st.number_input("Max vocab size", min_value=1000, max_value=50000, value=5000)
        sequence_length = st.number_input("Sequence length", min_value=3, max_value=50, value=10)
    with col2:
        embedding_dim = st.number_input("Embedding dim", min_value=32, max_value=256, value=100)
        lstm_units = st.number_input("LSTM units", min_value=32, max_value=256, value=128)
    
    epochs = st.sidebar.number_input("Training epochs", min_value=5, max_value=100, value=20)
    use_bidirectional = st.sidebar.checkbox("Use Bidirectional LSTM", value=True)

    # Tokenize and prepare data
    try:
        tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
        tokenizer.fit_on_texts([corpus_text])
        total_words = len(tokenizer.word_index) + 1
        
        sequences = tokenizer.texts_to_sequences([corpus_text])[0]
        
        # Create input sequences
        input_sequences = []
        for i in range(sequence_length, len(sequences)):
            seq = sequences[i-sequence_length:i+1]
            input_sequences.append(seq)
        
        if len(input_sequences) < 10:
            st.error("❌ Not enough data to create training sequences. Try a larger corpus or reduce sequence length.")
        else:
            input_sequences = np.array(input_sequences)
            X = input_sequences[:, :-1]
            y = tf.keras.utils.to_categorical(input_sequences[:, -1], num_classes=total_words)
            
            st.write(f"📊 Vocabulary size: {total_words}")
            st.write(f"📊 Training sequences: {X.shape[0]}")
            st.write(f"📊 Input shape: {X.shape}")
            
            # Build improved model
            model_text = Sequential([
                Embedding(total_words, embedding_dim, input_length=sequence_length),
                Bidirectional(LSTM(lstm_units, return_sequences=True)) if use_bidirectional else LSTM(lstm_units, return_sequences=True),
                Dropout(0.3),
                LSTM(lstm_units//2),
                Dropout(0.3),
                Dense(total_words//2, activation='relu'),
                Dense(total_words, activation='softmax')
            ])
            
            model_text.compile(
                loss='categorical_crossentropy', 
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy']
            )
            
            # Display model summary
            with st.expander("🔍 Model Architecture"):
                model_summary = []
                model_text.summary(print_fn=lambda x: model_summary.append(x))
                st.text('\n'.join(model_summary))

            # Training
            if st.sidebar.button("🚀 Train Next-Word Model"):
                with st.spinner("🔄 Training model... This may take a few minutes"):
                    # Split data for validation
                    split_idx = int(0.8 * len(X))
                    X_train, X_val = X[:split_idx], X[split_idx:]
                    y_train, y_val = y[:split_idx], y[split_idx:]
                    
                    history = model_text.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=64,
                        verbose=1
                    )
                    
                    # Save model and tokenizer
                    model_text.save("next_word_model.h5")
                    with open("tokenizer.pkl", "wb") as f:
                        pickle.dump(tokenizer, f)
                    
                    # Plot training history
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.plot(history.history['loss'], label='Training Loss')
                    ax1.plot(history.history['val_loss'], label='Validation Loss')
                    ax1.set_title('Model Loss')
                    ax1.legend()
                    
                    ax2.plot(history.history['accuracy'], label='Training Accuracy')
                    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
                    ax2.set_title('Model Accuracy')
                    ax2.legend()
                    
                    st.pyplot(fig)
                
                st.success("✅ Model trained and saved successfully!")

            # Prediction interface
            st.subheader("🔮 Next Word Prediction")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                seed_text = st.text_input("Enter seed text:", value="i am not able to", key="seed_input")
            with col2:
                temperature = st.slider("Creativity (temperature)", 0.1, 2.0, 0.7, 0.1)

            if st.button("🎯 Predict Next Words"):
                try:
                    # Load saved model if available
                    if os.path.exists("next_word_model.h5") and os.path.exists("tokenizer.pkl"):
                        loaded_model = load_model("next_word_model.h5")
                        loaded_tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
                    else:
                        loaded_model = model_text
                        loaded_tokenizer = tokenizer
                    
                    # Prepare input
                    seed_text_clean = preprocess_text(seed_text)
                    token_list = loaded_tokenizer.texts_to_sequences([seed_text_clean])[0]
                    token_list = pad_sequences([token_list], maxlen=sequence_length, padding='pre')
                    
                    # Predict
                    predicted_probs = loaded_model.predict(token_list, verbose=0)[0]
                    
                    # Get top predictions
                    top_indices, top_probs = top_k_from_preds(predicted_probs, k=5)
                    
                    st.write("**🏆 Top 5 Predictions:**")
                    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                        word = loaded_tokenizer.index_word.get(idx, "<UNK>")
                        st.write(f"{i+1}. **{word}** ({prob:.3f})")
                    
                    # Temperature sampling
                    sampled_idx = sample_with_temperature(predicted_probs, temperature)
                    sampled_word = loaded_tokenizer.index_word.get(sampled_idx, "<UNK>")
                    
                    st.write(f"**🎲 Creative suggestion:** {sampled_word}")
                    st.write(f"**💬 Complete sentence:** {seed_text} {sampled_word}")
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error processing corpus: {str(e)}")

# -------------------
# Program 2: Hit Song Prediction using LSTM
# -------------------
st.sidebar.header("🎵 Program 2: Hit Song Prediction")

# Dataset suggestions
with st.sidebar.expander("📊 Recommended Datasets"):
    st.write("""
    **Best Datasets for Hit Song Prediction:**
    1. **Spotify Tracks Dataset** (Kaggle) - Audio features + popularity
    2. **Million Song Dataset** - Comprehensive music data
    3. **Billboard Hot 100** - Chart performance data
    4. **Last.fm Dataset** - User listening patterns
    5. **MusicBrainz** - Rich music metadata
    
    **Required columns:** Audio features (tempo, energy, danceability, etc.) + target label
    """)

uploaded_song = st.sidebar.file_uploader("Upload music dataset (CSV)", type=["csv"], key="song_uploader")
use_sample_music_data = st.sidebar.checkbox("Generate sample music data for testing")

if use_sample_music_data:
    # Generate synthetic music data
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic music features
    music_data = {
        'danceability': np.random.beta(2, 2, n_samples),
        'energy': np.random.beta(2, 2, n_samples),
        'speechiness': np.random.beta(1, 5, n_samples),  # Most songs have low speechiness
        'acousticness': np.random.beta(1, 3, n_samples),
        'instrumentalness': np.random.beta(1, 10, n_samples),  # Most songs have vocals
        'liveness': np.random.beta(1, 5, n_samples),
        'valence': np.random.beta(2, 2, n_samples),
        'tempo': np.random.normal(120, 30, n_samples),  # BPM around 120
        'loudness': np.random.normal(-7, 3, n_samples),  # dB
        'duration_ms': np.random.normal(210000, 60000, n_samples),  # ~3.5 minutes
    }
    
    df_song = pd.DataFrame(music_data)
    
    # Create target based on realistic patterns (hits tend to have certain characteristics)
    hit_score = (
        df_song['danceability'] * 0.3 +
        df_song['energy'] * 0.25 +
        df_song['valence'] * 0.2 +
        (1 - df_song['acousticness']) * 0.15 +
        np.random.normal(0, 0.2, n_samples)  # Add noise
    )
    df_song['target'] = (hit_score > hit_score.quantile(0.7)).astype(int)  # Top 30% are hits
    
    st.info("Using synthetic music dataset for testing")

elif uploaded_song:
    df_song = pd.read_csv(uploaded_song)

if 'df_song' in locals():
    st.header("🎵 Hit Song Prediction")
    
    # Display dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Size", f"{len(df_song):,} songs")
    with col2:
        if 'target' in df_song.columns:
            hit_rate = df_song['target'].mean()
            st.metric("Hit Rate", f"{hit_rate:.1%}")
    with col3:
        st.metric("Features", len(df_song.select_dtypes(include=[np.number]).columns))
    
    # Show dataset preview
    st.write("**Dataset Preview:**")
    st.dataframe(df_song.head())
    
    # Feature selection
    numeric_cols = df_song.select_dtypes(include=[np.number]).columns.tolist()
    
    # Check for target column
    if 'target' not in df_song.columns:
        if 'hit' in df_song.columns:
            df_song['target'] = df_song['hit']
        elif 'popular' in df_song.columns:
            df_song['target'] = df_song['popular']
        else:
            st.error("❌ No target column found. Please ensure your dataset has a 'target', 'hit', or 'popular' column.")
            st.stop()
    
    # Remove target from features
    feature_cols = [col for col in numeric_cols if col not in ['target', 'hit', 'popular']]
    
    if len(feature_cols) < 1:
        st.error("❌ No numeric features found for prediction.")
        st.stop()
    
    # Feature selection interface
    selected_features = st.multiselect(
        "Select features for prediction:",
        feature_cols,
        default=feature_cols[:min(10, len(feature_cols))]  # Default to first 10 features
    )
    
    if not selected_features:
        st.warning("⚠️ Please select at least one feature.")
        st.stop()
    
    # Prepare data
    X = df_song[selected_features].values
    y = df_song['target'].values
    
    # Handle missing values
    if np.isnan(X).any():
        st.warning("⚠️ Found missing values, filling with median values.")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape for LSTM (samples, timesteps, features)
    # For time series, we'll create sequences
    sequence_length_song = st.sidebar.number_input("Sequence length for LSTM", min_value=1, max_value=10, value=1)
    
    if sequence_length_song > 1:
        # Create sequences (sliding window approach)
        X_sequences = []
        y_sequences = []
        for i in range(sequence_length_song, len(X_scaled)):
            X_sequences.append(X_scaled[i-sequence_length_song:i])
            y_sequences.append(y[i])
        X_lstm = np.array(X_sequences)
        y_lstm = np.array(y_sequences)
    else:
        # Reshape for single timestep
        X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        y_lstm = y
    
    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_lstm, y_lstm, test_size=0.2, random_state=42, stratify=y_lstm
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_lstm, y_lstm, test_size=0.2, random_state=42
        )
    
    st.write(f"📊 Training samples: {X_train.shape[0]}")
    st.write(f"📊 Test samples: {X_test.shape[0]}")
    st.write(f"📊 Input shape: {X_train.shape}")
    
    # Build LSTM model for hit prediction
    model_song = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model_song.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Display model architecture
    with st.expander("🔍 Hit Prediction Model Architecture"):
        model_summary = []
        model_song.summary(print_fn=lambda x: model_summary.append(x))
        st.text('\n'.join(model_summary))

    # Training
    song_epochs = st.sidebar.number_input("Training epochs (song)", min_value=10, max_value=200, value=50)
    
    if st.sidebar.button("🚀 Train Hit Prediction Model"):
        with st.spinner("🔄 Training hit prediction model..."):
            history = model_song.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=song_epochs,
                batch_size=32,
                verbose=1,
                class_weight={0: 1, 1: len(y_train)/(2*sum(y_train))}  # Balance classes
            )
            
            # Save model and scaler
            model_song.save("hit_song_model.h5")
            with open("song_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            
            # Plot training history
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss
            axes[0,0].plot(history.history['loss'], label='Training Loss')
            axes[0,0].plot(history.history['val_loss'], label='Validation Loss')
            axes[0,0].set_title('Model Loss')
            axes[0,0].legend()
            
            # Accuracy
            axes[0,1].plot(history.history['accuracy'], label='Training Accuracy')
            axes[0,1].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axes[0,1].set_title('Model Accuracy')
            axes[0,1].legend()
            
            # Precision
            axes[1,0].plot(history.history['precision'], label='Training Precision')
            axes[1,0].plot(history.history['val_precision'], label='Validation Precision')
            axes[1,0].set_title('Model Precision')
            axes[1,0].legend()
            
            # Recall
            axes[1,1].plot(history.history['recall'], label='Training Recall')
            axes[1,1].plot(history.history['val_recall'], label='Validation Recall')
            axes[1,1].set_title('Model Recall')
            axes[1,1].legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
        st.success("✅ Hit prediction model trained successfully!")
        
        # Evaluate model
        y_pred = (model_song.predict(X_test) > 0.5).astype(int)
        
        # Classification report
        st.write("**📈 Model Performance:**")
        report = classification_report(y_test, y_pred, output_dict=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{report['accuracy']:.3f}")
        with col2:
            st.metric("Precision (Hit)", f"{report['1']['precision']:.3f}")
        with col3:
            st.metric("Recall (Hit)", f"{report['1']['recall']:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    # Prediction interface
    st.subheader("🎯 Predict Song Success")
    
    if st.button("🔮 Predict Random Song"):
        try:
            # Load saved model if available
            if os.path.exists("hit_song_model.h5") and os.path.exists("song_scaler.pkl"):
                loaded_model = load_model("hit_song_model.h5")
                loaded_scaler = pickle.load(open("song_scaler.pkl", "rb"))
            else:
                loaded_model = model_song
                loaded_scaler = scaler
            
            # Get random sample
            random_idx = np.random.randint(0, len(X_test))
            sample = X_test[random_idx:random_idx+1]
            actual = y_test[random_idx]
            
            # Predict
            prediction_prob = loaded_model.predict(sample)[0][0]
            prediction_class = 1 if prediction_prob > 0.5 else 0
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Hit Probability", f"{prediction_prob:.3f}")
                st.metric("Prediction", "Hit 🎉" if prediction_class == 1 else "Not a Hit 😞")
            with col2:
                st.metric("Actual Result", "Hit 🎉" if actual == 1 else "Not a Hit 😞")
                st.metric("Correct?", "✅ Yes" if prediction_class == actual else "❌ No")
            
            # Show feature values
            st.write("**Song Features:**")
            sample_original = scaler.inverse_transform(sample.reshape(1, -1))[0]
            feature_df = pd.DataFrame({
                'Feature': selected_features,
                'Value': sample_original
            })
            st.dataframe(feature_df)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("## 💡 Tips for Better Results")
st.markdown("""
**For Next-Word Prediction:**
- Use domain-specific text (customer support, chat logs)
- Ensure sufficient training data (>1000 sentences)
- Experiment with sequence length and LSTM units
- Use temperature sampling for creative responses

**For Hit Song Prediction:**
- Include audio features (tempo, energy, danceability)
- Consider temporal features (release date, season)
- Balance your dataset (equal hits and non-hits)
- Use ensemble methods for better accuracy
""")