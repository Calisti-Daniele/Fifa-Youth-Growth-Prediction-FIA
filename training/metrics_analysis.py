from tensorflow.keras.models import load_model
import os
# Definisci il percorso dove salvare il modello
model_path = "../models/fia_model.keras"

# Carica il modello salvato per l'analisi
model = load_model(model_path)
print("Modello caricato con successo!")

# Valutazione del modello sul test set
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Errore assoluto medio (MAE) con il modello caricato: {mae}")

# Predizioni per l'analisi
y_pred_scaled = model.predict(X_test)

# Invertire la normalizzazione per interpretare i risultati
y_test_original = scaler_y.inverse_transform(y_test)
y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

# Calcolo delle metriche e analisi degli errori
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print(f"\nPerformance del modello caricato:")
print(f"- MAE: {mae:.4f}")
print(f"- MSE: {mse:.4f}")
print(f"- RMSE: {rmse:.4f}")
print(f"- R2-Score: {r2:.4f}")

# Continua con la tua analisi degli errori e visualizzazioni...
