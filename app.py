from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from fuzzywuzzy import process
import nest_asyncio
import os
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the uploaded dataset
uploaded_df = None

# Define the dependencies for the agent
@dataclass
class DatasetAnalysisDependencies:
    df: pd.DataFrame  # The dataset loaded as a Pandas DataFrame


# Define the result model for the agent
class DatasetAnalysisResult(BaseModel):
    analysis_result: str = Field(description="The result of the analysis or query")
    insights: Optional[List[str]] = Field(description="List of insights derived from the analysis")
    success: bool = Field(description="Whether the analysis was successful")


# Create the Dataset Analysis Agent with Gemini
model = GeminiModel('gemini-1.5-flash')  # Replace with your Gemini API key

dataset_analysis_agent = Agent(
    model=model,  # Use the Gemini model
    deps_type=DatasetAnalysisDependencies,
    result_type=DatasetAnalysisResult,
    system_prompt=(
        "You are a data analysis agent. Your job is to analyze any dataset "
        "and provide insights based on user queries. Use the tools provided to interact with the dataset."
    ),
)


# Function to clean the dataset
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by handling missing values and invalid data."""
    # Drop rows where all values are missing
    df.dropna(how="all", inplace=True)
    
    # Fill missing numeric values with 0
    for col in df.select_dtypes(include=["number"]).columns:
        df[col].fillna(0, inplace=True)
    
    # Fill missing string values with "Unknown"
    for col in df.select_dtypes(include=["object"]).columns:
        df[col].fillna("Unknown", inplace=True)
    
    return df


# Function to find the closest matching column name
def find_closest_column(df: pd.DataFrame, column_name: str) -> str:
    """Use fuzzy matching to find the closest matching column name."""
    choices = df.columns.tolist()
    match, score = process.extractOne(column_name, choices)
    if score >= 80:  # Only return a match if the score is above a threshold
        return match
    return None


# Add generic tools for the agent to interact with any dataset
@dataset_analysis_agent.tool
async def get_column_summary(
    ctx: RunContext[DatasetAnalysisDependencies], column: str
) -> Dict[str, Any]:
    """Retrieve summary statistics for a specific column in the dataset."""
    df = ctx.deps.df
    closest_column = find_closest_column(df, column)
    if not closest_column:
        return {"error": f"Column '{column}' not found in the dataset."}
    return df[closest_column].describe().to_dict()


@dataset_analysis_agent.tool
async def filter_dataset(
    ctx: RunContext[DatasetAnalysisDependencies], column: str, value: str
) -> List[dict]:
    """Filter the dataset based on a column and value."""
    df = ctx.deps.df
    closest_column = find_closest_column(df, column)
    if not closest_column:
        return [{"error": f"Column '{column}' not found in the dataset."}]
    filtered_data = df[df[closest_column] == value].to_dict(orient="records")
    return filtered_data


@dataset_analysis_agent.tool
async def sort_dataset(
    ctx: RunContext[DatasetAnalysisDependencies], column: str, ascending: bool = True, limit: int = 10
) -> List[dict]:
    """Sort the dataset based on a column and return the top results."""
    df = ctx.deps.df
    closest_column = find_closest_column(df, column)
    if not closest_column:
        return [{"error": f"Column '{column}' not found in the dataset."}]
    sorted_data = df.sort_values(by=closest_column, ascending=ascending).head(limit).to_dict(orient="records")
    return sorted_data


@dataset_analysis_agent.tool
async def group_by_column(
    ctx: RunContext[DatasetAnalysisDependencies], group_column: str, agg_column: str, agg_func: str = "mean"
) -> Dict[str, Any]:
    """Group the dataset by a column and apply an aggregation function to another column."""
    df = ctx.deps.df
    closest_group_column = find_closest_column(df, group_column)
    closest_agg_column = find_closest_column(df, agg_column)
    if not closest_group_column or not closest_agg_column:
        return {"error": f"One or more columns not found in the dataset."}
    grouped_data = df.groupby(closest_group_column)[closest_agg_column].agg(agg_func).to_dict()
    return grouped_data


@dataset_analysis_agent.tool
async def get_unique_values(
    ctx: RunContext[DatasetAnalysisDependencies], column: str
) -> List[str]:
    """Retrieve unique values from a specific column."""
    df = ctx.deps.df
    closest_column = find_closest_column(df, column)
    if not closest_column:
        return [f"Column '{column}' not found in the dataset."]
    return df[closest_column].unique().tolist()


@dataset_analysis_agent.tool
async def get_correlation(
    ctx: RunContext[DatasetAnalysisDependencies], column1: str, column2: str
) -> float:
    """Calculate the correlation between two columns."""
    df = ctx.deps.df
    closest_column1 = find_closest_column(df, column1)
    closest_column2 = find_closest_column(df, column2)
    if not closest_column1 or not closest_column2:
        return {"error": f"One or more columns not found in the dataset."}
    return float(df[closest_column1].corr(df[closest_column2]))


# Endpoint to upload a CSV file
@app.route("/upload", methods=["POST"])
def upload_csv():
    global uploaded_df
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        uploaded_df = pd.read_csv(file)
        uploaded_df = clean_dataset(uploaded_df)  # Clean the dataset
        return jsonify({"message": "File uploaded and dataset loaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Endpoint to query the dataset
@app.route("/query", methods=["POST"])
async def query_agent():
    global uploaded_df
    if uploaded_df is None:
        return jsonify({"error": "No dataset loaded. Please upload a file first."}), 400

    query = request.json.get("query")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        deps = DatasetAnalysisDependencies(df=uploaded_df.copy())
        result = await dataset_analysis_agent.run(query, deps=deps)
        return jsonify({
            "analysis_result": result.data.analysis_result,
            "insights": result.data.insights or [],
            "success": result.data.success,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    app.run()