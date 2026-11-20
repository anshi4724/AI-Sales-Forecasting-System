
import axios from "axios";

const API = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
  headers: {
    "Content-Type": "application/json",
  },
});

export const generateForecast = async (requestData) => {
  const response = await API.post("/predict/raw", requestData);
  return response.data;
};

export default API;
