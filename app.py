from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# Configure the LLM
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key="")

# Define a data model for the input
class InputData(BaseModel):
    input_data: Dict[str, Any]  # This will hold the actual data with possible missing fields
    all_field_info: Dict[str, str]  # This will describe expected fields and their types
    format_instructions: str  # Formatting instructions

def impute_missing_fields(input_data, all_field_info, format_instructions, model):
    # Updated prompt for better clarity and guidance to the model
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Given the following field information, fill in any missing values in the provided data. "
         "Use the format: {format_instructions}."),
        ("human", "Field Information: {all_field_info}, Data: {input_data}")
    ])
    
    chain = LLMChain(prompt=prompt, llm=model)
    
    try:
        # Pass the data into the chain
        result = chain.run({
            "input_data": input_data,
            "all_field_info": all_field_info,
            "format_instructions": format_instructions
        })
        
        # Return the parsed result from the LLM response
        return result
    except Exception as e:
        # If something goes wrong, return an error message
        return {"error": str(e)}

@app.post("/api/impute")
async def impute(data: InputData):
    try:
        # Extract the fields from the request
        input_data = data.input_data
        all_field_info = data.all_field_info
        format_instructions = data.format_instructions
        
        # Call the function to impute missing fields
        imputed_data = impute_missing_fields(input_data, all_field_info, format_instructions, model)
        
        # Return the imputed data
        return {"imputed_data": imputed_data}
    except Exception as e:
        # Handle errors and return a well-formed error response
        raise HTTPException(status_code=400, detail=str(e))
