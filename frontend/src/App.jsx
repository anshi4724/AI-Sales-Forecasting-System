
import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from "recharts";
import {
  LayoutDashboard, TrendingUp, ShoppingCart,
  Upload, BrainCircuit, CalendarDays, Loader2
} from "lucide-react";
import { generateForecast } from "./services/api";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [requestData, setRequestData] = useState(null);
  const [forecast, setForecast] = useState([]);
  const [forecastDates, setForecastDates] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const handleFileUpload = async (event) => {
    const selectedFile = event.target.files[0];

    if (!selectedFile) return;

    try {
      const content = await selectedFile.text();
      const json = JSON.parse(content);

      if (
        !Array.isArray(json.historical_data) ||
        !Array.isArray(json.future_dates)
      ) {
        throw new Error("JSON must contain historical_data and future_dates.");
      }

      if (
        json.historical_data.length !== 90 ||
        json.future_dates.length !== 16
      ) {
        throw new Error("Please upload 90 historical records and 16 future dates.");
      }

      setFile(selectedFile.name);
      setRequestData(json);
      setForecast([]);
      setForecastDates([]);
      setError("");
      setMessage("File uploaded successfully. Ready to forecast.");
    } catch (err) {
      setFile(null);
      setRequestData(null);
      setError(err.message || "Invalid JSON file.");
      setMessage("");
    }
  };

  const handleForecast = async () => {
    if (!requestData) {
      setError("Please upload your JSON file first.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setMessage("");

      const result = await generateForecast(requestData);

      setForecast(result.forecast);
      setForecastDates(requestData.future_dates);
      setMessage("16-day sales forecast generated successfully.");
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        "Unable to generate forecast."
      );
    } finally {
      setLoading(false);
    }
  };

  const averageForecast = forecast.length
    ? forecast.reduce((sum, value) => sum + value, 0) / forecast.length
    : 0;

  const totalForecast = forecast.reduce((sum, value) => sum + value, 0);

  const highestForecast = forecast.length
    ? Math.max(...forecast)
    : 0;

  const chartData = [
    ...(requestData?.historical_data || []).slice(-30).map((item) => ({
      date: item.date,
      actual: item.sales,
      predicted: null,
    })),
    ...forecast.map((value, index) => ({
      date: forecastDates[index],
      actual: null,
      predicted: value,
    })),
  ];

  const formatNumber = (value) =>
    new Intl.NumberFormat("en-IN", {
      maximumFractionDigits: 2,
    }).format(value);

  return (
    <div className="dashboard">

      <aside className="sidebar">
        <div className="brand">
          <BrainCircuit size={30} />
          <span>SalesAI</span>
        </div>

        <div className="nav-heading">MENU</div>

        <div className="nav-item active">
          <LayoutDashboard size={19} />
          Dashboard
        </div>

        <div className="nav-item">
          <TrendingUp size={19} />
          Forecasting
        </div>

        <div className="nav-item">
          <ShoppingCart size={19} />
          Sales Analysis
        </div>

        <div className="sidebar-footer">
          <div className="status-dot"></div>
          AI Model Connected
        </div>
      </aside>

      <main className="main-content">

        <header className="topbar">
          <div>
            <h2>Sales Forecasting Dashboard</h2>
            <p>AI-powered sales prediction system</p>
          </div>

          <div className="model-status">
            <span className="status-dot"></span>
            LSTM Model
          </div>
        </header>

        <section className="welcome-section">
          <div>
            <h1>Sales Overview</h1>
            <p>Analyze historical sales and predict future demand.</p>
          </div>
          <CalendarDays size={32} />
        </section>

        <section className="upload-section">
          <div className="upload-icon">
            <Upload size={25} />
          </div>

          <div className="upload-info">
            <h3>Upload Sales Data</h3>
            <p>
              Select your JSON file containing 90 historical records
              and 16 future dates.
            </p>
            {file && <span className="file-name">Selected: {file}</span>}
          </div>

          <label className="upload-button">
            Choose JSON File
            <input
              type="file"
              accept=".json,application/json"
              onChange={handleFileUpload}
              hidden
            />
          </label>
        </section>

        {message && <div className="success-message">{message}</div>}
        {error && <div className="error-message">{error}</div>}

        <div className="generate-row">
          <button
            className="generate-button"
            onClick={handleForecast}
            disabled={loading || !requestData}
          >
            {loading ? (
              <>
                <Loader2 className="spin" size={19} />
                Generating Forecast...
              </>
            ) : (
              <>
                <BrainCircuit size={19} />
                Generate 16-Day Forecast
              </>
            )}
          </button>
        </div>

        <section className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon purple">
              <TrendingUp />
            </div>
            <p>Forecast Horizon</p>
            <h2>{forecast.length || 16} Days</h2>
            <span>Prediction period</span>
          </div>

          <div className="stat-card">
            <div className="stat-icon blue">
              <ShoppingCart />
            </div>
            <p>Average Predicted Sales</p>
            <h2>{formatNumber(averageForecast)}</h2>
            <span>Per day</span>
          </div>

          <div className="stat-card">
            <div className="stat-icon green">
              <TrendingUp />
            </div>
            <p>Total Predicted Sales</p>
            <h2>{formatNumber(totalForecast)}</h2>
            <span>Across forecast period</span>
          </div>

          <div className="stat-card">
            <div className="stat-icon orange">
              <BrainCircuit />
            </div>
            <p>Highest Predicted Sales</p>
            <h2>{formatNumber(highestForecast)}</h2>
            <span>Single day</span>
          </div>
        </section>

        <section className="chart-section">
          <div className="section-heading">
            <div>
              <h3>Sales Trend & Forecast</h3>
              <p>Historical sales and predicted sales</p>
            </div>
            <span className="chart-label">30 Historical Days + 16 Forecast Days</span>
          </div>

          <div className="chart-container">
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 11 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    name="Historical Sales"
                    stroke="#6366f1"
                    strokeWidth={2}
                    dot={false}
                    connectNulls={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    name="Predicted Sales"
                    stroke="#10b981"
                    strokeWidth={3}
                    dot={{ r: 3 }}
                    connectNulls={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state">
                Upload your sales data and generate a forecast to see the chart.
              </div>
            )}
          </div>
        </section>

        <section className="table-section">
          <div className="section-heading">
            <div>
              <h3>16-Day Sales Forecast</h3>
              <p>Daily predicted sales generated by the LSTM model</p>
            </div>
          </div>

          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Day</th>
                  <th>Forecast Date</th>
                  <th>Predicted Sales</th>
                </tr>
              </thead>

              <tbody>
                {forecast.length > 0 ? (
                  forecast.map((value, index) => (
                    <tr key={index}>
                      <td>Day {index + 1}</td>
                      <td>{forecastDates[index]}</td>
                      <td className="sales-value">
                        {formatNumber(value)}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="3" className="empty-table">
                      No forecast available. Upload data and click Generate Forecast.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <footer>
          AI Sales Forecasting System | Powered by PyTorch LSTM and FastAPI
        </footer>

      </main>
    </div>
  );
}

export default App;
